"""
配置文件
集中管理所有超参数和路径配置
"""

import torch

# ==================== 设备配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 数据配置 ====================
DATA_ROOT = "./data"           # 数据集存储根目录
BATCH_SIZE = 64                # 批次大小
NUM_WORKERS = 0                # 数据加载线程数 (Windows建议设为0)

# MNIST数据集标准化参数
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# ==================== 模型配置 ====================
NUM_CLASSES = 10               # 分类类别数 (数字0-9)
INPUT_CHANNELS = 1             # 输入通道数 (灰度图=1, RGB=3)

# ==================== 训练配置 ====================
EPOCHS = 10                    # 训练轮数
LEARNING_RATE = 0.001          # 学习率
WEIGHT_DECAY = 1e-4            # 权重衰减 (L2正则化)

# ==================== 路径配置 ====================
CHECKPOINT_DIR = "./checkpoints"   # 模型检查点保存目录
RESULTS_DIR = "./results"          # 结果保存目录 (图表等)
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"

# 确保目录存在
import os
for directory in [DATA_ROOT, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


def print_config():
    """打印当前配置信息"""
    print("=" * 50)
    print("项目配置信息")
    print("=" * 50)
    print(f"设备: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"模型保存路径: {BEST_MODEL_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
