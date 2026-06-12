# Vanilla GAN vs DCGAN — MNIST 生成对比

课程作业项目：在 MNIST 手写数字数据集上实现并对比 **Vanilla GAN**（Goodfellow et al., 2014）与 **DCGAN**（Radford et al., 2015），用于课堂演示两种架构在相同训练流程下的生成效果差异。

## 项目结构

```
GAN/
├── models/
│   ├── vanilla_gan.py   # 全连接 MLP 生成器/判别器，28×28
│   └── dcgan.py         # 卷积/转置卷积，32×32
├── data.py              # MNIST 数据加载（共用）
├── train.py             # 统一训练入口
├── compare.py           # 并排对比两个已训练模型
├── smoke_test.py        # 冒烟测试（不下载数据）
├── utils.py             # 工具函数
├── requirements.txt
└── outputs/             # 训练输出（运行后生成）
```

## 环境要求

- Python 3.8+
- **NVIDIA GPU + CUDA**（推荐；代码会自动检测并使用 `cuda`）
- 依赖见 `requirements.txt`

### 安装

```bash
pip install -r requirements.txt
```

验证 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 快速开始

### 1. 冒烟测试（可选，无需下载数据）

确认两个模型前向/反向传播正常：

```bash
python smoke_test.py
```

### 2. 训练（GPU 推荐参数）

在 GPU 上 MNIST 规模较小，可适当增大 batch、多训几轮以提升生成质量。

```bash
# DCGAN（推荐先训，效果通常更好）
python train.py --model dcgan --epochs 50 --batch_size 256

# Vanilla GAN（对照组）
python train.py --model vanilla --epochs 50 --batch_size 256
```

**默认超参**（与 DCGAN 论文一致）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latent_dim` | 100 | 噪声向量维度 |
| `lr` | 2e-4 | Adam 学习率 |
| `betas` | (0.5, 0.999) | Adam 动量 |
| `batch_size` | 128 | 可用 `--batch_size 256` 加速 GPU 训练 |
| `epochs` | 30 | 展示建议 50 epoch |

**训练输出**（以 `dcgan` 为例）：

```
outputs/dcgan/
├── generator.pth          # 生成器权重（compare.py 需要）
├── final_samples.png      # 最终生成样本网格
├── loss_curve.png         # G/D 损失曲线
└── samples/
    └── epoch_001.png ...  # 每 epoch 固定噪声下的样本演变
```

首次训练会自动下载 MNIST 到 `./data`。

### 3. 生成对比图

两个模型都训练完成后：

```bash
python compare.py
```

输出：`outputs/comparison.png`（**同一批噪声**下 Vanilla GAN 与 DCGAN 并排对比）。

可选参数：

```bash
python compare.py --n 64 --latent_dim 100 --out_root ./outputs
```

## 两种模型对比（汇报要点）

| | Vanilla GAN | DCGAN |
|---|-------------|-------|
| 结构 | 全连接 MLP | 转置卷积 G + 卷积 D |
| 图像尺寸 | 28×28 | 32×32（MNIST 放大） |
| 池化 | 无 | 用 stride 卷积替代池化 |
| BatchNorm | 生成器有 | G/D 均有（D 首层除外） |
| 典型效果 | 较模糊、易模式崩塌 | 更清晰、结构更稳定 |

训练目标相同：标准 GAN 极小极大博弈，损失为 `BCELoss`，交替更新 D 与 G。

## 演示日 Checklist

- [ ] `outputs/vanilla/generator.pth` 与 `outputs/dcgan/generator.pth` 已存在
- [ ] `outputs/comparison.png` 已生成
- [ ] 本地可运行 `python compare.py`（无需现场重新训练）
- [ ] PPT 中放入：`comparison.png`、两条 `loss_curve.png`、可选 `epoch_*.png` 演变图

## 常见问题

**Q: 训练时显示 `使用设备: cpu`？**  
安装带 CUDA 的 PyTorch，并确认 `nvidia-smi` 正常。见 [PyTorch 安装页](https://pytorch.org/get-started/locally/)。

**Q: `compare.py` 提示找不到权重？**  
先分别运行 `train.py --model vanilla` 与 `train.py --model dcgan`。

**Q: DCGAN 生成器输入噪声形状？**  
`compare.py` 与 `train.py` 传入 `(N, 100)`；`dcgan.Generator.forward` 会自动 reshape 为 `(N, 100, 1, 1)`。

## 参考文献

- Goodfellow, I., et al. (2014). *Generative Adversarial Nets.* NeurIPS.
- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* ICLR.

## 汇报材料

PPT 逐页大纲见 [PRESENTATION_OUTLINE.md](./PRESENTATION_OUTLINE.md)。
