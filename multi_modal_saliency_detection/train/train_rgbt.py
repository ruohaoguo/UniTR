import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import numpy as np
import torch
import torch.nn.functional as F

from datetime import datetime
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')

from options import opt
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from models.unitr_swin import build_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


# Dataset Configs
rgb_root = opt.train_rgbt_rgb
t_root = opt.train_rgbt_t
gt_root = opt.train_rgbt_gt
edge_root = opt.train_rgbt_edge

test_rgb_root = opt.test_rgbt_rgb_root
test_t_root = opt.test_rgbt_t_root
test_gt_root = opt.test_rgbt_gt_root


# Save Configs
save_path = "./run_models/"
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Log Configs
logging.basicConfig(filename = save_path + 'UniTR_rgbt_swin.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("UniTR_rgbt_swin")


# Build Model
model = build_model()
if (opt.load is not None):
    model.load_pre(opt.load)
    print('load model from ', opt.load)
model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


# Load Data
print('load data...')
train_loader = get_loader(rgb_root, gt_root, t_root, edge_root, batchsize=opt.batch_size, trainsize=opt.image_size)
test_loader = test_dataset(test_rgb_root, test_gt_root, test_t_root, opt.image_size)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batch_size, opt.image_size, opt.clip, opt.decay_rate, opt.load, save_path, opt.decay_epoch))


# Loss
CE = torch.nn.BCEWithLogitsLoss()


# TensorboardX
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# Train
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    for i, (rgb, gt, t, edge) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        rgb = rgb.cuda()
        gt = gt.cuda()
        t = t.repeat(1, 3, 1, 1).cuda()
        sal1, sal2 = model(rgb, t)

        sal1_loss = CE(sal1, gt)
        sal2_loss = CE(sal2, gt)
        loss = sal1_loss + sal2_loss
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        step += 1
        epoch_step += 1
        loss_all += loss.data
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        if i % 100 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal1_loss:{:4f} ||sal2_loss:{:4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         optimizer.state_dict()['param_groups'][0]['lr'], sal1_loss.data, sal2_loss.data))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal1_loss:{:4f} ||sal2_loss:{:4f}'.
                format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], sal1_loss.data, sal2_loss.data))
            writer.add_scalar('Loss', loss.data, global_step=step)

    loss_all /= epoch_step
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    if epoch >= 150 and epoch % 10 == 0:
        torch.save(model.state_dict(), save_path + 'UniTR_rgbt_swin_epoch_{}.pth'.format(epoch))


# Test
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            rgb, gt, t, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            rgb = rgb.cuda()
            t = t.repeat(1, 3, 1, 1).cuda()

            s, e = model(rgb, t)

            s = F.upsample(s, size=gt.shape, mode='bilinear', align_corners=False)
            s = s.sigmoid().data.cpu().numpy().squeeze()
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            mae_sum += np.sum(np.abs(s - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'UniTR_rgbt_swin_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        if epoch > 120:
            test(test_loader, model, epoch, save_path)
