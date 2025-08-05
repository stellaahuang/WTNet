import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from math import log10
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from dataloader import MyDataset
from model import Network

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

log_path = "./train.log"
set_logger(log_path)
logger = logging.getLogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_function = nn.MSELoss()
batch_size = 32
epochs = 300
lr = 5e-4
# result_path = "/home/syhuang/all_in_one"
# os.makedirs(result_path, exist_ok=True)

def plot_curve(data, title, ylabel, path):
    plt.figure()
    plt.title(title, fontsize=18)
    plt.plot(data, label="train")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()

def plot_all_losses(img_loss_list, mask_loss_list, A_loss_list, beta_loss_list, path):
    plt.figure()
    plt.title("Loss Curves", fontsize=18)
    plt.plot(img_loss_list, label="Image Loss")
    plt.plot(mask_loss_list, label="Mask Loss")
    plt.plot(A_loss_list, label="A Loss")
    plt.plot(beta_loss_list, label="Beta Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def PSNR(img1, img2, data_range=1.):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def train_epoch(model, dataloader, optimizer, best_psnr):
    model.train()
    total_img_loss = 0
    total_mask_loss = 0
    total_A_loss = 0
    total_beta_loss = 0
    total_psnr = 0

    with tqdm(dataloader, unit="Batch", desc="Train") as tqdm_loader:
        for index, (deg_img, gt_img, gt_mask, depth, gt_A, gt_beta) in enumerate(tqdm_loader):
            deg_img = deg_img.to(device)
            gt_img = gt_img.to(device)
            gt_mask = gt_mask.to(device)
            depth = depth.to(device)
            gt_A = torch.tensor(gt_A, dtype=torch.float32).to(device)
            gt_beta = torch.tensor(gt_beta, dtype=torch.float32).to(device)

            refine, A, beta, mask = model(deg_img, gt_img, depth)

            img_loss = loss_function(refine, deg_img)
            mask_loss = loss_function(mask, gt_mask)
            A_loss = loss_function(A, gt_A)
            beta_loss = loss_function(beta, gt_beta)
            loss = img_loss + mask_loss + A_loss + beta_loss

            loss /= 2
            loss.backward()
            if (index + 1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_img_loss += img_loss.item()
            total_mask_loss += mask_loss.item()
            total_A_loss += A_loss.item()
            total_beta_loss += beta_loss.item()

            psnr = PSNR(refine, deg_img).item()
            total_psnr += psnr

            avg_img_loss = total_img_loss / (index + 1)
            avg_mask_loss = total_mask_loss / (index + 1)
            avg_A_loss = total_A_loss / (index + 1)
            avg_beta_loss = total_beta_loss / (index + 1)
            avg_psnr = total_psnr / (index + 1)

            tqdm_loader.set_postfix(img_loss=avg_img_loss, mask_loss=avg_mask_loss, psnr=avg_psnr)

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "./best_model.pkl")

    logger.info(f"--Train-- img_loss: {avg_img_loss:.6f}, mask_loss: {avg_mask_loss:.6f}, A_loss: {avg_A_loss:.6f}, beta_loss: {avg_beta_loss:.6f}, PSNR: {avg_psnr:.6f}")
    return avg_psnr, avg_img_loss, avg_mask_loss, avg_A_loss, avg_beta_loss

def train(model, train_dataloader, optimizer):
    best_psnr = -100
    psnr_list = []
    img_loss_list = []
    mask_loss_list = []
    A_loss_list = []
    beta_loss_list = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}:")
        psnr, img_loss, mask_loss, A_loss, beta_loss = train_epoch(model, train_dataloader, optimizer, best_psnr)
        best_psnr = max(best_psnr, psnr)

        psnr_list.append(psnr)
        img_loss_list.append(img_loss)
        mask_loss_list.append(mask_loss)
        A_loss_list.append(A_loss)
        beta_loss_list.append(beta_loss)

    curve_path = os.path.join(os.getcwd(), "curves")
    os.makedirs(curve_path, exist_ok=True)

    plot_curve(psnr_list, "PSNR Curve", "PSNR", os.path.join(curve_path, "psnr_curve.jpg"))

    plot_all_losses(img_loss_list, mask_loss_list, A_loss_list, beta_loss_list,
                    os.path.join(curve_path, "loss_curve.jpg"))

    return psnr_list, img_loss_list, mask_loss_list, A_loss_list, beta_loss_list

if __name__ == "__main__":
    deg_root = "/YOUR/SYNTHETIC/DERADED/IMAGES/ROOT"
    gt_root = "/YOUR/SYNTHETIC/GT/IMAGES/ROOT"
    mask_root = "/YOUR/SYNTHETIC/MASK/ROOT"
    depth_root = "/YOUR/SYNTHETIC/DEPTH/ROOT"

    train_dataset = MyDataset(deg_root, gt_root, mask_root, depth_root)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Network(device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_psnr, train_img_loss, train_mask_loss, train_A_loss, train_beta_loss = train(model, train_dataloader, optimizer)
