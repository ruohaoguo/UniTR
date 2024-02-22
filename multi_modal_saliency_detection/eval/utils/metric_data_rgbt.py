import os
from PIL import Image
import torchvision.transforms as transforms

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    # RGB-T
    def load_data(self, s='VT5000'):
        image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.png'))
        if s == 'VT5000':
            gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        else:
            gt = self.binary_loader(os.path.join(self.gt_root, self.img_list[self.index] + '.jpg'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

