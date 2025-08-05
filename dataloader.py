from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import torch
import numpy as np

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

class MyDataset(Dataset):
    def __init__(self, deg_root, gt_root, mask_root, depth_root, trans_root):
        self.deg_root = deg_root
        self.gt_root = gt_root
        self.mask_root = mask_root
        self.depth_root = depth_root
        self.trans_root = trans_root
        
        self.deg_paths = sorted([os.path.join(self.deg_root, f) for f in os.listdir(self.deg_root)])
        self.toTensor = ToTensor()

    def __len__(self):
        return len(self.deg_paths)

    def __getitem__(self, index):
        deg_path = self.deg_paths[index]
        deg_filename = os.path.basename(deg_path)
        
        deg_parts = deg_filename.split('-')
        deg_type = deg_parts[0]
        A = 0.0
        beta = 0.0
        if deg_type == "haze":
            deg_index = deg_parts[1].split('_')[0]
            _, A_val, beta_val = deg_filename[:-4].split('_')
            A = float(A_val)
            beta = float(beta_val)
        else:
            deg_index = deg_parts[1].split('.')[0]
            A = 0.0
            beta = 0.0

        gt_path = os.path.join(self.gt_root, f"{deg_type}_clean-{deg_index}.png")
        mask_path = os.path.join(self.mask_root, f"{deg_type}_mask-{deg_index}.png")
        depth_path = os.path.join(self.depth_root, f"{deg_type}_depth-{deg_index}.npy")
        trans_path = os.path.join(self.trans_root, f"{os.path.splitext(deg_filename)[0]}.npy")

        deg_img = self.toTensor(Image.open(deg_path).convert('RGB'))
        gt_img = self.toTensor(Image.open(gt_path).convert('RGB'))
        mask = self.toTensor(Image.open(mask_path).convert('L'))
        depth = self.toTensor(np.load(depth_path).astype(np.float32))
        trans = self.toTensor(np.load(trans_path).astype(np.float32))

        return deg_img, gt_img, mask, depth, A, beta, trans