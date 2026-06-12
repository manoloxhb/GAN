"""对比脚本:把 Vanilla GAN 与 DCGAN 的生成结果并排展示,用于课程演示。

前提:已分别训练好两个模型(outputs/vanilla 与 outputs/dcgan 下有 generator.pth)。

用法:
    python compare.py                # 加载已训练权重并排出图
    python compare.py --n 64         # 指定生成样本数
"""
import os
import argparse
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils import get_device
from models import vanilla_gan, dcgan


def load_generator(model_name, ckpt_path, latent_dim, device):
    if model_name == "vanilla":
        G = vanilla_gan.Generator(latent_dim=latent_dim)
    else:
        G = dcgan.Generator(latent_dim=latent_dim)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.to(device).eval()
    return G


def make_grid_img(generator, noise, nrow):
    with torch.no_grad():
        imgs = generator(noise)
    grid = vutils.make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
    return grid.permute(1, 2, 0).cpu().numpy()


def main(args):
    device = get_device()
    nrow = int(args.n ** 0.5)
    # 用同一批噪声,直观对比两个模型对相同输入的生成效果
    noise = torch.randn(args.n, args.latent_dim, device=device)

    vanilla_ckpt = os.path.join(args.out_root, "vanilla", "generator.pth")
    dcgan_ckpt = os.path.join(args.out_root, "dcgan", "generator.pth")

    panels = []
    if os.path.exists(vanilla_ckpt):
        G = load_generator("vanilla", vanilla_ckpt, args.latent_dim, device)
        panels.append(("Vanilla GAN (MLP)", make_grid_img(G, noise, nrow)))
    else:
        print(f"未找到 {vanilla_ckpt},请先训练: python train.py --model vanilla")

    if os.path.exists(dcgan_ckpt):
        G = load_generator("dcgan", dcgan_ckpt, args.latent_dim, device)
        panels.append(("DCGAN (Conv)", make_grid_img(G, noise, nrow)))
    else:
        print(f"未找到 {dcgan_ckpt},请先训练: python train.py --model dcgan")

    if not panels:
        print("没有可用的模型权重,退出。")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    fig.suptitle("Vanilla GAN vs DCGAN — MNIST 生成结果对比", fontsize=18)
    out_path = os.path.join(args.out_root, "comparison.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"对比图已保存: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="对比 Vanilla GAN 与 DCGAN")
    p.add_argument("--n", type=int, default=64, help="生成样本数(最好是平方数)")
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--out_root", default="./outputs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
