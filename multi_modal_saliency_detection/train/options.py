import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

parser.add_argument('--image_size', type=int, default=224, help='training dataset size')
parser.add_argument('--load', type=str, default='../pre_models/swin_base_patch4_window7_224_22k.pth')


# 1.RGBD: train_2185
parser.add_argument('--train_rgbd_2185_rgb', type=str, default='../datasets/rgbd/train_2185/train_images/', help='the training rgb images root')
parser.add_argument('--train_rgbd_2185_d', type=str, default='../datasets/rgbd/train_2185/train_depth/', help='the training depth images root')
parser.add_argument('--train_rgbd_2185_gt', type=str, default='../datasets/rgbd/train_2185/train_masks/', help='the training gt images root')
parser.add_argument('--train_rgbd_2185_edge', type=str, default='../datasets/rgbd/train_2185/train_edges/', help='the training edge images root')

# 2.RGBD: train_2985
parser.add_argument('--train_rgbd_2985_rgb', type=str, default='../datasets/rgbd/train_2985/train_images/', help='the training rgb images root')
parser.add_argument('--train_rgbd_2985_d', type=str, default='../datasets/rgbd/train_2985/train_depth/', help='the training depth images root')
parser.add_argument('--train_rgbd_2985_gt', type=str, default='../datasets/rgbd/train_2985/train_masks/', help='the training gt images root')
parser.add_argument('--train_rgbd_2985_edge', type=str, default='../datasets/rgbd/train_2985/train_edges/', help='the training edge images root')

# 3.RGBT: train
parser.add_argument('--train_rgbt_rgb', type=str, default='../datasets/VTDataset/train/RGB/', help='the training rgb images root')
parser.add_argument('--train_rgbt_t', type=str, default='../datasets/VTDataset/train/T/', help='the training t images root')
parser.add_argument('--train_rgbt_gt', type=str, default='../datasets/VTDataset/train/GT/', help='the training gt images root')
parser.add_argument('--train_rgbt_edge', type=str, default='../datasets/VTDataset/train/Edge/', help='the training edge images root')

# 4.RGBD: test
parser.add_argument('--test_rgbd_rgb_root', type=str, default='../datasets/rgbd/test/DES/RGB/', help='the test gt images root')
parser.add_argument('--test_rgbd_d_root', type=str, default='../datasets/rgbd/test/DES/depth/', help='the test gt images root')
parser.add_argument('--test_rgbd_gt_root', type=str, default='../datasets/rgbd/test/DES/GT/', help='the test gt images root')

# 5.RGBT: test
parser.add_argument('--test_rgbt_rgb_root', type=str, default='../datasets/VTDataset/test/VT821/RGB/', help='the test gt images root')
parser.add_argument('--test_rgbt_t_root', type=str, default='../datasets/VTDataset/test/VT821/T/', help='the test gt images root')
parser.add_argument('--test_rgbt_gt_root', type=str, default='../datasets/VTDataset/test/VT821/GT/', help='the test gt images root')


parser.add_argument('--save_rgbd_2185_path', type=str, default='./run_models/', help='the path to save models and logs')
parser.add_argument('--save_rgbd_2985_path', type=str, default='./run_models/', help='the path to save models and logs')
parser.add_argument('--save_rgbt_path', type=str, default='./run_models/', help='the path to save models and logs')
opt = parser.parse_args()
