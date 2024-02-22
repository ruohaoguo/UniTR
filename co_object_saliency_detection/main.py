import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import time
import argparse
import torch
import queue
import threading
from pycocotools import coco
from model.model_image import build_model
from utils.tools import custom_print
from utils.data_processed import train_data_producer
from train import train

torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', default='./pre_models/swin_base_patch4_window7_224_22k.pth')
    parser.add_argument('--anno_path', default='./datasets/train/coco/annotations/instances_train2017.json')
    parser.add_argument('--npy_path', default='./utils/cat2imgid_dict4000.npy')
    parser.add_argument('--img_size', default=224)
    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--lr_de', default=20000)
    parser.add_argument('--epochs', default=60000)
    parser.add_argument('--bs', default=8, help='batch size')
    parser.add_argument('--gs', default=5, help='group size')
    parser.add_argument('--log_interval', default=100, help='log interval')
    parser.add_argument('--val_interval', default=1000, help='val interval')
    args = parser.parse_args()

    data_train_path = './datasets/train/coco/train2017/'
    data_val_path = ['./datasets/test/CoCA', './datasets/test/CoSal2015', './datasets/test/CoSOD3k']

    pretrain_path = args.pretrain_path
    coco_item = coco.COCO(annotation_file=args.anno_path)
    npy = args.npy_path

    # project config
    project_name = 'UniTR_cosod_swin'
    device = torch.device('cuda:0')
    img_size = args.img_size
    lr = args.lr
    lr_de = args.lr_de
    epochs = args.epochs
    batch_size = args.bs
    group_size = args.gs
    log_interval = args.log_interval
    val_interval = args.val_interval

    # create log dir and txt
    log_root = './logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    log_txt_file = os.path.join(log_root, project_name + '_log.txt')
    custom_print(project_name, log_txt_file, 'w')

    # create model save dir
    models_root = './run_models'
    if not os.path.exists(models_root):
        os.makedirs(models_root)
    models_train_last = os.path.join(models_root, project_name + '_last.pth')
    models_train_best_coca = os.path.join(models_root, project_name + '_best_coca.pth')
    models_train_best_cosal = os.path.join(models_root, project_name + '_best_casal.pth')
    models_train_best_cosod = os.path.join(models_root, project_name + '_best_cosod.pth')

    net = build_model().to(device)
    net.train()
    net.load_pre(pretrain_path)
    net=torch.nn.DataParallel(net)

    q = queue.Queue(maxsize=40)

    p1 = threading.Thread(target=train_data_producer,
                          args=(coco_item, data_train_path, npy, q, batch_size, group_size, img_size))
    p2 = threading.Thread(target=train_data_producer,
                          args=(coco_item, data_train_path, npy, q, batch_size, group_size, img_size))
    p3 = threading.Thread(target=train_data_producer,
                          args=(coco_item, data_train_path, npy, q, batch_size, group_size, img_size))
    p1.start()
    p2.start()
    p3.start()
    time.sleep(2)

    train(project_name, models_root, net, device, q, log_txt_file, data_val_path,
          models_train_best_coca, models_train_best_cosal, models_train_best_cosod, models_train_last,
          lr, lr_de, epochs, log_interval, val_interval)
