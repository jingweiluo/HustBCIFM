import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import random
import time
import matplotlib.pyplot as plt

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

def to_tensor(array):
    return torch.from_numpy(array).float()

def save_loss_curve(epoch_losses, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(save_dir, f"loss_curve_{timestamp}.png")

    plt.figure()
    plt.plot(np.arange(1, len(epoch_losses) + 1), epoch_losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)