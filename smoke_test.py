"""冒烟测试:不下载 MNIST,用随机张量跑若干 batch,验证两个模型的前向/反向传播。

用法:
    python smoke_test.py
    python smoke_test.py --batches 3 --batch_size 16
"""
import argparse
import os

import torch
import torch.nn as nn

from models import dcgan, vanilla_gan
from train import build_models
from utils import get_device, save_sample_grid


def run_smoke(model_name, latent_dim, batch_size, num_batches, device):
    """模拟 train.py 的训练循环,用随机图像替代真实数据。"""
    G, D, image_size = build_models(model_name, latent_dim, device)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()

    for step in range(num_batches):
        real_imgs = torch.randn(batch_size, 1, image_size, image_size, device=device)
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # ---- 判别器 ----
        opt_D.zero_grad()
        d_real = D(real_imgs)
        assert d_real.shape == (batch_size, 1), f"D(real) 形状错: {d_real.shape}"

        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(noise)
        assert fake_imgs.shape == (batch_size, 1, image_size, image_size), (
            f"G 输出形状错: {fake_imgs.shape}"
        )

        d_fake = D(fake_imgs.detach())
        assert d_fake.shape == (batch_size, 1), f"D(fake) 形状错: {d_fake.shape}"

        loss_D = criterion(d_real, real_label) + criterion(d_fake, fake_label)
        loss_D.backward()
        opt_D.step()

        # ---- 生成器 ----
        opt_G.zero_grad()
        d_fake_for_g = D(fake_imgs)
        loss_G = criterion(d_fake_for_g, real_label)
        loss_G.backward()
        opt_G.step()

        print(
            f"  [{model_name}] batch {step + 1}/{num_batches} "
            f"Loss_D={loss_D.item():.4f} Loss_G={loss_G.item():.4f}"
        )

    # 固定噪声前向 + 保存样本网格
    G.eval()
    with torch.no_grad():
        samples = G(torch.randn(16, latent_dim, device=device))
    out_path = os.path.join("outputs", "_smoke", f"{model_name}.png")
    save_sample_grid(samples, out_path, nrow=4)
    print(f"  [{model_name}] 样本已保存: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Vanilla GAN / DCGAN 冒烟测试")
    p.add_argument("--batches", type=int, default=2, help="每个模型跑的 batch 数")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--latent_dim", type=int, default=100)
    args = p.parse_args()

    device = get_device()
    print(f"设备: {device} | batches={args.batches} | batch_size={args.batch_size}")

    for name in ("vanilla", "dcgan"):
        run_smoke(name, args.latent_dim, args.batch_size, args.batches, device)

    print("全部冒烟测试通过。")


if __name__ == "__main__":
    main()
