"""
数据加载与预处理模块
负责加载MNIST数据集并进行必要的预处理操作
"""

import torch
from torchvision import datasets, transforms
from config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, MNIST_MEAN, MNIST_STD


def get_transform(train=True):
    """
    定义图像预处理变换
    
    Args:
        train: 是否为训练集的训练模式(可添加数据增强)
        
    Returns:
        transform: torchvision.transforms.Compose对象
    """
    if train:
        # 训练集: 可以添加数据增强
        transform = transforms.Compose([
            transforms.ToTensor(),                                    # 转为张量 [0,255] -> [0,1]
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))         # 标准化
        ])
    else:
        # 测试集: 只做基本预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        ])
    
    return transform


def get_data_loaders(batch_size=None):
    """
    获取训练集和测试集的数据加载器
    
    Args:
        batch_size: 批次大小, 若None则使用配置文件中的默认值
        
    Returns:
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=get_transform(train=True)
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=get_transform(train=False)
    )
    
    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,              # 训练集打乱数据
        num_workers=NUM_WORKERS
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,             # 测试集不需要打乱
        num_workers=NUM_WORKERS
    )
    
    print(f"训练集大小: {len(train_dataset)} 张图片")
    print(f"测试集大小: {len(test_dataset)} 张图片")
    print(f"批次大小: {batch_size}")
    print(f"每epoch迭代次数(训练): {len(train_loader)}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # 查看一批数据的形状
    images, labels = next(iter(train_loader))
    print(f"\n批次数据形状:")
    print(f"  图片: {images.shape}")      # [batch, channels, height, width]
    print(f"  标签: {labels.shape}")      # [batch]
    print(f"  标签示例: {labels[:10].tolist()}")
