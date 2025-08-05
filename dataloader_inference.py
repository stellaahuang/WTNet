from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np
import random
import cv2

seed = 666
random.seed(seed)

def transform(img):
    img_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])(img)
    return img_trans

class MyDataset(Dataset):
    def __init__(self, deg_root, gt_root, target_root, target_depth_root):
        self.deg_root = deg_root
        self.gt_root = gt_root
        self.target_root = target_root
        self.target_depth_root = target_depth_root
        self.deg_paths = [os.path.join(self.deg_root, f) for f in sorted(os.listdir(self.deg_root))]
        self.gt_paths = [os.path.join(self.gt_root, f) for f in sorted(os.listdir(self.gt_root))]
        self.target_paths = [os.path.join(self.target_root, f) for f in sorted(os.listdir(self.target_root)) if f != '.DS_Store']
        self.target_depth_paths = [os.path.join(self.target_depth_root, f) for f in sorted(os.listdir(self.target_depth_root)) if f != '.DS_Store']

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, index):
        random_index = random.randint(0, len(self.deg_paths) - 1)
        deg_path = self.deg_paths[random_index]
        gt_path = self.gt_paths[random_index]
        target_path = self.target_paths[index]
        target_depth_path = self.target_depth_paths[index]

        deg_img = Image.open(deg_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')
        target_depth = np.load(target_depth_path)

        deg_img = transform(deg_img)
        gt_img = transform(gt_img)
        target = transform(target)
        target_depth = cv2.resize(target_depth, (256, 256), interpolation=cv2.INTER_LINEAR)

        return deg_img, gt_img, target, target_depth
