"""统一训练入口,可切换 Vanilla GAN 与 DCGAN。

用法:
    python train.py --model vanilla --epochs 30
    python train.py --model dcgan   --epochs 30

两个模型共用同一套训练循环(标准 GAN 的极小极大博弈),
只是网络结构和输入图像尺寸不同,方便课程上直接对比。
"""
import os
import argparse
import torch
import torch.nn as nn

from data import get_mnist_loader
from utils import weights_init_dcgan, save_sample_grid, plot_losses, get_device
from models import vanilla_gan, dcgan


def build_models(model_name, latent_dim, device):
    """根据名称构建生成器/判别器,返回 (G, D, image_size)。"""
    if model_name == "vanilla":
        G = vanilla_gan.Generator(latent_dim=latent_dim).to(device)
        D = vanilla_gan.Discriminator().to(device)
        image_size = vanilla_gan.IMAGE_SIZE
    elif model_name == "dcgan":
        G = dcgan.Generator(latent_dim=latent_dim).to(device)
        D = dcgan.Discriminator().to(device)
        # DCGAN 用论文推荐的权重初始化
        G.apply(weights_init_dcgan)
        D.apply(weights_init_dcgan)
        image_size = dcgan.IMAGE_SIZE
    else:
        raise ValueError(f"未知模型: {model_name}")
    return G, D, image_size


def train(args):
    device = get_device()
    print(f"使用设备: {device} | 模型: {args.model}")

    G, D, image_size = build_models(args.model, args.latent_dim, device)
    loader = get_mnist_loader(args.batch_size, image_size, args.data_root)

    criterion = nn.BCELoss()  # 二元交叉熵,标准 GAN 损失
    # DCGAN 论文推荐 lr=2e-4, betas=(0.5, 0.999)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 固定噪声,用于观察同一批 z 在训练过程中的生成演变
    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    out_dir = os.path.join(args.out_root, args.model)
    sample_dir = os.path.join(out_dir, "samples")
    g_losses, d_losses = [], []

    for epoch in range(args.epochs):
        for i, (real_imgs, _) in enumerate(loader):
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)
            # 标签:真实=1,生成=0
            real_label = torch.ones(bs, 1, device=device)
            fake_label = torch.zeros(bs, 1, device=device)

            # ---- 训练判别器:最大化 log D(x) + log(1 - D(G(z))) ----
            opt_D.zero_grad()
            d_real = D(real_imgs)
            loss_real = criterion(d_real, real_label)

            noise = torch.randn(bs, args.latent_dim, device=device)
            fake_imgs = G(noise)
            d_fake = D(fake_imgs.detach())  # detach 避免更新 G
            loss_fake = criterion(d_fake, fake_label)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ---- 训练生成器:最大化 log D(G(z)) ----
            opt_G.zero_grad()
            d_fake_for_g = D(fake_imgs)  # 重新过判别器,这次保留梯度
            loss_G = criterion(d_fake_for_g, real_label)  # 希望判别器判成真
            loss_G.backward()
            opt_G.step()

            if i % args.log_interval == 0:
                g_losses.append(loss_G.item())
                d_losses.append(loss_D.item())
                print(
                    f"[{epoch+1}/{args.epochs}][{i}/{len(loader)}] "
                    f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}"
                )

        # 每个 epoch 用固定噪声保存一张生成样本网格
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
        G.train()
        save_sample_grid(
            samples, os.path.join(sample_dir, f"epoch_{epoch+1:03d}.png")
        )

    # 训练结束:保存最终样本、loss 曲线、模型权重
    save_sample_grid(samples, os.path.join(out_dir, "final_samples.png"))
    plot_losses(
        g_losses, d_losses, os.path.join(out_dir, "loss_curve.png"),
        title=f"{args.model.upper()} Training Loss",
    )
    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pth"))
    print(f"完成。输出保存在: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Vanilla GAN / DCGAN on MNIST")
    p.add_argument("--model", choices=["vanilla", "dcgan"], default="dcgan")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_root", default="./outputs")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
