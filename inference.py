import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import logging
import sys
from thop import profile
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from math import log10
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from dataloader_inference import MyDataset
from model import Network
import cv2
import time 

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_function = nn.MSELoss()
batch_size = 1
epochs = 1
lr = 1e-5
factor = 64

def PSNR(img1, img2, data_range=1.):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def save_img(image, path):
    image = image.squeeze(0)
    img = TF.to_pil_image(image)
    img.save(path)

def pad_img(img, h, w):
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    padded_img = F.pad(img, (0, padw, 0, padh), 'reflect')
    return padded_img

def test_epoch(model, dataloader):
    with torch.no_grad():
        total_psnr = 0
        total_time = 0
        total_images = 0
        with tqdm(dataloader, unit = "Batch", desc = "Inference") as tqdm_loader:
            for index, (deg_img, gt_img, target, target_depth) in enumerate(tqdm_loader):
                # forward
                deg_img = deg_img.to(device)
                gt_img = gt_img.to(device)
                target = target.to(device)
                target_depth = target_depth.to(device)
                torch.cuda.synchronize()
                start_time = time.time()

                result, A, beta, target_mask = model.inference(gt_img, target, target_depth)
                torch.cuda.synchronize()
                end_time = time.time()

                batch_time = end_time - start_time
                total_time += batch_time
                total_images += deg_img.size(0)

                # calculate avg loss and psnr
                psnr = PSNR(result, deg_img)
                psnr = psnr.detach().item()
                total_psnr += psnr
                avg_psnr = total_psnr / (index + 1)

                # print progress
                tqdm_loader.set_postfix(psnr = avg_psnr)

                # save results
                changed_path = os.path.join("/ssddisk/syhuang/ablation_parameter/350", "mask_change_rain")
                os.makedirs(changed_path, exist_ok=True)

                rain_path = os.path.join(changed_path, "rain")
                gt_path = os.path.join(changed_path, "gt")
                target_path = os.path.join(changed_path, "target")
                os.makedirs(rain_path, exist_ok=True)
                os.makedirs(gt_path, exist_ok=True)
                os.makedirs(target_path, exist_ok=True)

                save_img(result, os.path.join(rain_path, f'result_{index + 1}.jpg'))
                save_img(gt_img, os.path.join(gt_path, f'clean_{index + 1}.jpg'))
                save_img(target, os.path.join(target_path, f'target_{index + 1}.jpg'))
                # save_img(target_mask, os.path.join(target_path, f'target_mask_{index + 1}.jpg'))
        avg_time_per_image = total_time / total_images
        print(f"Average inference time per image: {avg_time_per_image * 1000:.2f} ms")

def transform(img):
    img_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])(img)
    return img_trans

def test_for_one(model, clean_img_root, target_img_root, target_depth_root):
    clean_img = Image.open(clean_img_root).convert('RGB')
    target_img = Image.open(target_img_root).convert('RGB')
    target_depth = np.load(target_depth_root)

    clean_img = transform(clean_img).to(device).unsqueeze(0)
    target_img = transform(target_img).to(device).unsqueeze(0)
    target_depth = torch.from_numpy(cv2.resize(target_depth, (256, 256), interpolation=cv2.INTER_LINEAR)).to(device).unsqueeze(0)

    result, A, beta, target_mask = model.inference(clean_img, target_img, target_depth)
    save_img(result, './result.png')
    save_img(clean_img, './clean_img.png')
    save_img(target_img, './target_img.png')


def test(model, test_dataloader):
    for epoch in range(epochs):
        test_epoch(model, test_dataloader)


if __name__ == "__main__":
    deg_root = "/ssddisk/syhuang/All_in_one/degraded"
    gt_root = "/ssddisk/syhuang/All_in_one/gt"
    target_root = "/ssddisk/syhuang/WeatherStream/rain/deg"
    target_depth_root = "/ssddisk/syhuang/WeatherStream/rain/depth"

    test_dataset = MyDataset(deg_root, gt_root, target_root, target_depth_root)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last = True)

    model = Network(device).to(device)
    model.load_state_dict(torch.load('/ssddisk/syhuang/ablation_parameter/best_model_350.pkl', map_location=device))

        # === 加入 FLOPs 計算 ===
    # dummy_clean = torch.randn(1, 3, 256, 256).to(device)
    # dummy_target = torch.randn(1, 3, 256, 256).to(device)
    # dummy_depth = torch.randn(1, 256, 256).to(device)

    # # 確保 model.inference 是一個 nn.Module.forward-like function
    # def forward_pass(*inputs):
    #     return model.inference(*inputs)[0]  # 只取 result

    # flops, params = profile(model, inputs=(dummy_clean, dummy_target, dummy_depth), verbose=False)
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')


    test(model, test_dataloader)

    # test_for_one(model, clean_img_root, target_img_root, target_depth_root)
