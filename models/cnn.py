"""
CNN模型定义文件
定义用于MNIST手写数字识别的卷积神经网络架构
"""

import torch.nn as nn


class CNN(nn.Module):
    """
    卷积神经网络模型用于MNIST手写数字识别
    
    模型结构:
        Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> FC -> ReLU -> FC
    """
    
    def __init__(self, num_classes=10, input_channels=1):
        """
        初始化CNN模型
        
        Args:
            num_classes: 分类数量 (默认10,对应数字0-9)
            input_channels: 输入通道数 (灰度图为1, RGB为3)
        """
        super(CNN, self).__init__()
        
        # 第一个卷积块: 提取底层特征(边缘、线条等)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # 第二个卷积块: 提取高层特征(形状、组合特征等)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # 全连接层: 进行分类
        # 经过两个池化层后,特征图大小为 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout防止过拟合
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, channels, height, width]
            
        Returns:
            output: 分类 logits [batch_size, num_classes]
        """
        # 第一个卷积块
        x = self.pool1(self.relu1(self.conv1(x)))
        
        # 第二个卷积块
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # 展平特征图
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def get_model(num_classes=10, input_channels=1):
    """
    获取CNN模型的工厂函数
    
    Args:
        num_classes: 分类数量
        input_channels: 输入通道数
        
    Returns:
        model: CNN实例
    """
    return CNN(num_classes=num_classes, input_channels=input_channels)


if __name__ == "__main__":
    # 测试模型结构
    import torch
    model = get_model()
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
