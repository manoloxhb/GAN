"""通用工具:权重初始化、样本网格保存、训练曲线绘制。"""
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def weights_init_dcgan(m):
    """DCGAN 论文推荐的权重初始化:均值 0、标准差 0.02 的正态分布。"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_sample_grid(images, path, nrow=8):
    """把一批生成图像([-1,1])保存成网格 PNG。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 反归一化回 [0,1] 便于显示
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(ndarr, cmap="gray")
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1, dpi=120)
    plt.close()


def plot_losses(g_losses, d_losses, path, title="Training Loss"):
    """绘制生成器/判别器 loss 曲线。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(g_losses, label="Generator", alpha=0.8)
    plt.plot(d_losses, label="Discriminator", alpha=0.8)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
