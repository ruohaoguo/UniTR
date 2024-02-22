import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')

from models.unitr_swin import build_model
# from models.unitr_res import build_model
from data.data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224, help='testing size')
# parser.add_argument('--image_size', type=int, default=256, help='testing size')
parser.add_argument('--test_path',type=str,default='../datasets/rgbd/test/')
parser.add_argument('--model_path',type=str,default='../save_models/unitr_rgbd_swin.pth')
# parser.add_argument('--model_path',type=str,default='../save_models/unitr_rgbd_res.pth')
opt = parser.parse_args()
dataset_path = opt.test_path
model_path = opt.model_path
test_datasets = ['SIP','NJU2K','NLPR','STERE']


model = build_model()
model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()
for dataset in test_datasets:
    print(dataset)
    save_path = '../predictions/unitr_rgbd_swin/' + dataset + '/'
    # save_path = '../predictions/unitr_rgbd_res/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rgb_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    t_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(rgb_root, gt_root, t_root, opt.image_size)
    for i in range(test_loader.size):
        rgb, gt, t, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        rgb = rgb.cuda()
        t = t.repeat(1, 3, 1, 1).cuda()
        res, _ = model(rgb, t)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        cv2.imwrite(save_path + name, res * 255)
