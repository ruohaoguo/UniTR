import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import os
import torch
from model.model_video import build_model
from utils.tools import custom_print
from train import train_finetune
import argparse

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # train_val_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./save_models/unitr_cosod_swin.pth',
                        help="restore checkpoint")
    parser.add_argument('--img_size', default=224, help="size of input image")
    parser.add_argument('--lr', default=1e-5, help="learning rate")
    parser.add_argument('--lr_de', default=20000, help="learning rate decay")
    parser.add_argument('--batch_size', default=4, help="batch size")
    parser.add_argument('--group_size', default=5, help="group size")
    parser.add_argument('--epochs', default=100000, help="epoch")
    args = parser.parse_args()

    dataset_train_path = './datasets/train/DAVIS_FBMS/'
    dataset_val_path = ['./datasets/test/FBMS']

    # project config
    project_name = 'UniTR_vsod_swin'
    device = torch.device('cuda:0')
    img_size = args.img_size
    lr = args.lr
    lr_de = args.lr_de
    epochs = args.epochs
    batch_size = args.batch_size
    group_size = args.group_size
    log_interval = 100
    val_interval = 1000

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
    models_train_last = os.path.join(models_root, project_name + '_last_ft.pth')
    models_train_best_fbms = os.path.join(models_root, project_name + '_best_ft_davis.pth')
    models_train_best = os.path.join(models_root, project_name + '_best_ft.pth')

    # continute load checkpoint
    model_path = args.model
    gpu_id = 'cuda:0'
    device = torch.device(gpu_id)
    net = build_model()

    for p in net.cls_1[0].parameters():
      p.requires_grad=False
    for p in net.cls_2[0].parameters():
      p.requires_grad=False

    net = net.to(device)
    net = torch.nn.DataParallel(net)
    state_dict = torch.load(model_path, map_location=gpu_id)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.train()

    train_finetune(net, dataset_train_path, device, batch_size, log_txt_file, dataset_val_path, models_train_best_fbms,
                   models_train_last, lr, lr_de, epochs, log_interval, val_interval)
