"""DCGAN (Radford et al., 2015)。

深度卷积 GAN:用转置卷积做上采样生成图像,用卷积做下采样判别。
遵循论文要点:
  - 用 stride 卷积/转置卷积替代池化层
  - 生成器和判别器都用 BatchNorm
  - 生成器隐层用 ReLU,输出层用 Tanh
  - 判别器隐层用 LeakyReLU
  - 去掉全连接层
输入图像为 32x32(MNIST 放大),便于 4 次 2 倍下采样。
"""
import torch
import torch.nn as nn


IMAGE_SIZE = 32
CHANNELS = 1


class Generator(nn.Module):
    """转置卷积生成器:z (N,latent,1,1) -> (N,1,32,32)。"""

    def __init__(self, latent_dim=100, feature_g=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            # 输入 z: (latent_dim, 1, 1) -> (feature_g*4, 4, 4)
            nn.ConvTranspose2d(latent_dim, feature_g * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            # (feature_g*4, 4, 4) -> (feature_g*2, 8, 8)
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            # (feature_g*2, 8, 8) -> (feature_g, 16, 16)
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # (feature_g, 16, 16) -> (1, 32, 32)
            nn.ConvTranspose2d(feature_g, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        # z 形状需为 (N, latent_dim, 1, 1)
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)


class Discriminator(nn.Module):
    """卷积判别器:(N,1,32,32) -> 真/假概率。"""

    def __init__(self, feature_d=64):
        super().__init__()
        self.model = nn.Sequential(
            # (1, 32, 32) -> (feature_d, 16, 16),首层不用 BatchNorm
            nn.Conv2d(CHANNELS, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d, 16, 16) -> (feature_d*2, 8, 8)
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d*2, 8, 8) -> (feature_d*4, 4, 4)
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_d*4, 4, 4) -> (1, 1, 1)
            nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.model(img)
        return out.view(img.size(0), 1)  # 展平成 (N, 1)
