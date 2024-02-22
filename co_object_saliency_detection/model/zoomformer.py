import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)
    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x[:, :1, :])
        kv = self.to_kv(x[:, 1:, :]).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ZoomFormer(nn.Module):
    def __init__(self, g_h1=4, g_w1=4, g_h2=2, g_w2=2, dim=512, depth=2, heads=8, dropout=0.):
        super().__init__()

        self.g_h1 = g_h1
        self.g_w1 = g_w1
        self.g_h2 = g_h2
        self.g_w2 = g_w2

        self.unfold1 = nn.Unfold(kernel_size=4, stride=4, padding=0)
        self.unfold2 = nn.Unfold(kernel_size=2, stride=2, padding=0)
        self.unfold3 = nn.Unfold(kernel_size=1, stride=1, padding=0)

        self.fold3 = nn.Fold(output_size=7,  kernel_size=1, stride=1, padding=0)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 3, padding=1))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Cross_Attention(dim, heads=heads, dropout=dropout),
                PreNorm(dim=dim),
                get_ff()]))

    def forward(self, x1, x2, x3):
        b, c, h, w = x3.shape

        x1_unfold = self.unfold1(x1)
        x2_unfold = self.unfold2(x2)
        x3_unfold = self.unfold3(x3)

        x1 = rearrange(x1_unfold, 'b (c n) hw -> (b hw) n c', c=c)
        x2 = rearrange(x2_unfold, 'b (c n) hw -> (b hw) n c', c=c)
        x3 = rearrange(x3_unfold, 'b (c n) hw -> (b hw) n c', c=c)

        x = torch.cat((x3, x2, x1), dim=1)

        for (attn, norm, conv) in self.layers:
            x = norm(attn(x)) + x[:, :1, :]
            x = rearrange(x, '(b h w) n c -> b c h w n', h=h, w=w)
            x = x.squeeze(dim=-1)
            x = conv(x) + x
            x = x.unsqueeze(dim=-1)
            x = rearrange(x, 'b c h w n -> (b h w) n c')
            x = torch.cat((x, x2, x1), dim=1)

        x3 = x[:, :1, :]
        x3 = rearrange(x3, '(b h w) n c -> b (c n) (h w)', h=h, w=w)
        x3 = self.fold3(x3)

        return x3


