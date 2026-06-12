"""MNIST 数据加载工具,供 Vanilla GAN 与 DCGAN 共用。"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loader(batch_size=128, image_size=28, data_root="./data"):
    """返回 MNIST 的 DataLoader。

    像素被归一化到 [-1, 1],与生成器输出层使用的 Tanh 激活相匹配。

    Args:
        batch_size: 每个 batch 的样本数。
        image_size: 输出图像边长。Vanilla GAN 用 28,DCGAN 用 32(便于卷积下采样)。
        data_root: 数据集存放目录,首次运行会自动下载。
    """
    transform_list = []
    # DCGAN 需要 32x32,这里把 28x28 放大;Vanilla GAN 保持 28x28。
    if image_size != 28:
        transform_list.append(transforms.Resize(image_size))
    transform_list += [
        transforms.ToTensor(),  # 转到 [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # 映射到 [-1, 1]
    ]
    transform = transforms.Compose(transform_list)

    dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下设 0 避免多进程问题
        drop_last=True,
    )
    return loader
