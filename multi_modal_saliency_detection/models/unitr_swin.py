import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .swin_backbone import SwinTransformer
from .zoomformer import ZoomFormer
from .coformer import CoFormer


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def concat_r():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
    return layers

def concat_1():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    return layers

def incr_channel():
    layers = []
    layers += [nn.Conv2d(128, 512, 3, 1, 1)]
    layers += [nn.Conv2d(256, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(1024, 512, 3, 1, 1)]
    return layers

def incr_channel2():
    layers = []
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    return layers

def norm(x, dim):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    normed = x / torch.sqrt(squared_norm)
    return normed

def fuse_hsp(x, p, group_size=5):
    t = torch.zeros(group_size, x.size(1))
    for i in range(x.size(0)):
        tmp = x[i, :]
        if i == 0:
            nx = tmp.expand_as(t)
        else:
            nx = torch.cat(([nx, tmp.expand_as(t)]), dim=0)
    nx = nx.view(x.size(0) * group_size, x.size(1), 1, 1)
    y = nx.expand_as(p)
    return y


class Model(nn.Module):
    def __init__(self, incr_channel, incr_channel2, concat_r, concat_1):
        super(Model, self).__init__()

        self.base = SwinTransformer(img_size=224, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7)
        self.incr_channel1 = nn.ModuleList(incr_channel)
        self.incr_channel2 = nn.ModuleList(incr_channel2)

        self.h_transformer = ZoomFormer(dim=512, depth=2, heads=8)
        self.mm_transformer = CoFormer(dim=512, depth=4, heads=4, mlp_dim=782, group=2)

        self.cls_1 = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512), nn.ReLU())
        self.cls_2 = nn.Sequential(nn.Linear(512, 78), nn.Sigmoid())

        self.attns_conv = conv3x3_bn_relu(512, 256)
        self.attns = nn.Conv2d(256, 24, 3, 1, 1)

        self.bases_conv = conv3x3_bn_relu(512, 256)
        self.bases = nn.Conv2d(256, 24, 3, 1, 1)

        self.concat4 = nn.ModuleList(concat_r)
        self.concat3 = nn.ModuleList(concat_r)
        self.concat2 = nn.ModuleList(concat_r)
        self.concat1 = nn.ModuleList(concat_1)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv_concat = nn.Conv2d(2, 2, 3, 1, 1)

    def forward(self, x ,d):
        x = torch.cat((x.unsqueeze(dim=1), d.unsqueeze(dim=1)), dim=1)
        x = rearrange(x, 'b g c h w -> (b g) c h w')

        p = self.base(x)
        p.pop(4)

        # increase the channel
        newp = list()
        for k in range(len(p)):
            np = self.incr_channel1[k](p[k])
            np = self.incr_channel2[k](np)
            newp.append(self.incr_channel2[4](np))

        p3_hf = self.h_transformer(newp[1], newp[2], newp[3])
        mmf = self.mm_transformer(p3_hf)

        y4 = p3_hf
        for k in range(len(self.concat4)):
            y4 = self.concat4[k](y4)

        y3 = y4
        for k in range(len(self.concat3)):
            y3 = self.concat3[k](y3)

        y2 = y3
        for k in range(len(self.concat2)):
            y2 = self.concat2[k](y2)

        y1 = newp[0]
        for k in range(len(self.concat1)):
            y1 = self.concat1[k](y1)
            if k == 1:
                y1 = y1 + y2

        y = y1

        attns_features = self.attns_conv(mmf)
        attns = self.attns(attns_features)

        bases_features = self.bases_conv(y)
        bases = self.bases(bases_features)

        merge_features = self.merge_bases(bases, attns)
        pred_mask_logits = self.up4(merge_features)
        pred_mask_logits = rearrange(pred_mask_logits, '(b g) c h w -> b (c g) h w', g=2)
        pred_mask_logits = self.conv_concat(pred_mask_logits)
        sal = pred_mask_logits[:, :1, :, :]
        edge = pred_mask_logits[:, 1:, :, :]

        return sal, edge


    def merge_bases(self, bases, attns):
        B, N, H, W = bases.size()
        attns = F.interpolate(attns, (H, W), mode="bilinear").softmax(dim=1)
        masks_preds = (bases * attns).sum(dim=1).unsqueeze(dim=1)
        return masks_preds

    def load_pre(self, pre_model):
        self.base.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"SwinTransformer loading pre_model {pre_model}")


# build the whole network
def build_model():
    return Model(incr_channel(), incr_channel2(), concat_r(), concat_1())


