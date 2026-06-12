"""Vanilla GAN (Goodfellow et al., 2014)。

最原始的 GAN:生成器和判别器都用全连接层(MLP)。
输入是把 28x28 的 MNIST 图像展平成 784 维向量。
"""
import torch
import torch.nn as nn


IMAGE_SIZE = 28
IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE  # 784


class Generator(nn.Module):
    """全连接生成器:噪声向量 z -> 784 维图像向量。"""

    def __init__(self, latent_dim=100, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        def block(in_feat, out_feat):
            # 每层:线性 + BatchNorm + LeakyReLU,逐步放大特征维度
            return [
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.model = nn.Sequential(
            *block(latent_dim, hidden_dim),
            *block(hidden_dim, hidden_dim * 2),
            *block(hidden_dim * 2, hidden_dim * 4),
            nn.Linear(hidden_dim * 4, IMAGE_DIM),
            nn.Tanh(),  # 输出范围 [-1, 1],与归一化后的数据匹配
        )

    def forward(self, z):
        img = self.model(z)
        # 把 784 维向量重塑成图像 (N, 1, 28, 28)
        return img.view(img.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)


class Discriminator(nn.Module):
    """全连接判别器:784 维图像向量 -> 真/假概率。"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMAGE_DIM, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Dropout 缓解判别器过强
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # 输出 [0,1] 概率
        )

    def forward(self, img):
        # 把图像展平成向量再送入全连接层
        flat = img.view(img.size(0), -1)
        return self.model(flat)
