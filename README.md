# 🔢 手写数字识别系统 (MNIST CNN)

基于 **PyTorch** 的卷积神经网络手写数字识别系统，采用模块化架构设计，注重 API 易用性、代码可维护性与生产级部署能力。

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| **测试集准确率** | ≥ **99.0%** (目标 90%) |
| **模型参数量** | ~**440K** (轻量化) |
| **训练时间 (CPU)** | ~**10 epochs / 5-8 分钟** |
| **推理延迟 (单张)** | **< 10ms** |
| **输入尺寸** | 28×28 灰度图 |

### 模型架构

```
Input(1,28,28) → Conv2d(32,3×3) → ReLU → MaxPool(2×2)
              → Conv2d(64,3×3)   → ReLU → MaxPool(2×2)
              → Flatten(64×7×7=3136) → FC(128) → Dropout(0.5) → FC(10)
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于 GPU 加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 一键训练与评估

```bash
# 使用默认配置运行
python main.py

# 自定义超参数
python main.py --batch-size 128 --epochs 20 --lr 0.0005
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch-size` | int | 64 (config.py) | 批次大小 |
| `--epochs` | int | 10 (config.py) | 训练轮数 |
| `--lr` | float | 0.001 (config.py) | 学习率 |
| `--no-save` | flag | False | 跳过模型保存 |
| `--no-vis` | flag | False | 跳过可视化输出 |
| `--load-model` | str | None | 加载已有模型进行评估 |

---

## 🛠️ API 易用性

### 核心模块导入（统一入口）

```python
from utils import get_data_loaders, Trainer, plot_training_history, visualize_predictions
from models.cnn import get_model
```

### 极简使用示例

```python
# ===== 完整训练流程仅需 6 行核心代码 =====
train_loader, test_loader = get_data_loaders(batch_size=64)
model = get_model(num_classes=10, input_channels=1)

trainer = Trainer(model, lr=0.001)
history = trainer.train(train_loader, test_loader, epochs=10)
loss, accuracy = trainer.evaluate(test_loader)

# 可视化
plot_training_history(history)
visualize_predictions(model, test_loader, num_images=16)
```

### 模块化 API 设计

| 模块 | 功能 | 关键接口 |
|------|------|----------|
| `models/cnn.py` | CNN 模型定义 | `get_model()` 工厂函数 |
| `utils/dataset.py` | 数据加载与预处理 | `get_data_loaders()`, `get_transform()` |
| `utils/trainer.py` | 训练与评估封装 | `Trainer` 类 (train/evaluate/save/load) |
| `utils/visualize.py` | 结果可视化 | `plot_training_history()`, `visualize_predictions()`, `plot_confusion_matrix()` |
| `config.py` | 集中配置管理 | 所有超参数与路径常量 |

---

## 📁 项目结构

```
project/
├── main.py                 # 主程序入口（命令行解析 + 流程编排）
├── config.py               # 集中式配置管理（单一数据源）
├── requirements.txt        # 依赖声明
├── models/
│   ├── cnn.py             # CNN 模型定义（支持自定义输入/类别数）
│   └── __init__.py
├── utils/
│   ├── __init__.py         # 统一导出接口 (__all__ 声明)
│   ├── dataset.py          # 数据加载器工厂
│   ├── trainer.py          # Trainer 类（训练/评估/保存/加载）
│   └── visualize.py        # 可视化工具集
├── checkpoints/            # 模型权重存储（自动创建）
│   └── best_model.pth      # 最佳模型检查点
├── data/                   # MNIST 数据集缓存
└── results/                # 训练结果图表输出
```

---

## ✅ 错误处理机制

### 多层容错设计

| 场景 | 处理策略 | 代码位置 |
|------|----------|----------|
| **可选依赖缺失** | try-except 优雅降级，提示安装命令 | `main.py:95-100` |
| **设备自适应** | 自动检测 CUDA/CPU，无 GPU 时静默回退 | `config.py:9` |
| **目录不存在** | 启动时自动创建必要目录 | `config.py:36-37` |
| **参数校验** | argparse 类型检查 + 默认值兜底 | `main.py:19-36` |
| **模型加载安全** | `map_location` 兼容 CPU/GPU 设备迁移 | `trainer.py:197` |

### 示例：可选依赖优雅降级 (`main.py:95-100`)

```python
try:
    plot_confusion_matrix(model, test_loader, show_plot=True)
except ImportError:
    print("  [跳过] 需要安装 scikit-learn 和 seaborn 才能绘制混淆矩阵")
    print("  安装命令: pip install scikit-learn seaborn")
```

---

## 📝 日志记录

### 结构化训练日志输出

程序执行过程中提供清晰的阶段性日志：

```
==================================================
项目配置信息
==================================================
设备: cuda / cpu
批次大小: 64
训练轮数: 10
学习率: 0.001
模型保存路径: ./checkpoints/best_model.pth
==================================================

[Step 1] 加载MNIST数据集...
----------------------------------------
训练集大小: 60000 张图片
测试集大小: 10000 张图片
批次大小: 64
每epoch迭代次数(训练): 938

[Step 2] 构建CNN模型...
----------------------------------------
模型结构: CNN(...)

[Step 3] 初始化训练器...
----------------------------------------

开始训练, 共 10 个epoch
------------------------------------------------------------
Epoch [1/10] | Train Loss: 0.1325 Acc: 96.02% || Val Loss: 0.0823 Acc: 97.45%
  >> 保存最佳模型! 准确率: 97.45%
Epoch [2/10] | Train Loss: 0.0421 Acc: 98.67% || Val Loss: 0.0512 Acc: 98.32%
...

[Step 6] 最终评估报告
========================================
最终测试集表现:
  - 测试损失: 0.0356
  - 测试准确率: 99.12%

✓ 成功达到目标准确率 (90.0%)!
```

### 训练历史记录

`Trainer` 内置完整的训练历史追踪：

```python
trainer.history = {
    'train_loss': [...],       # 每个 epoch 的训练损失
    'train_accuracy': [...],   # 每个 epoch 的训练准确率
    'val_loss': [...],         # 每个 epoch 的验证损失
    'val_accuracy': [...]      # 每个 epoch 的验证准确率
}
```

---

## 🏗️ 代码可维护性

### 设计原则

| 原则 | 实现方式 |
|------|----------|
| **关注点分离** | Model / Data / Train / Visualize 四层解耦 |
| **配置集中化** | 单一 `config.py` 管理所有超参数和路径 |
| **依赖注入** | Trainer 支持自定义 criterion/optimizer/device |
| **工厂模式** | `get_model()` 封装模型实例化细节 |
| **开放封闭** | CNN 类通过构造参数扩展，无需修改源码 |

### 扩展指南

```python
# 1. 修改网络结构：编辑 models/cnn.py 的 CNN 类
# 2. 调整超参数：只需修改 config.py，全局生效
# 3. 替换优化器：Trainer(model, optimizer=torch.optim.SGD(...))
# 4. 自定义数据集：实现兼容 DataLoader 接口即可
# 5. 添加新的可视化函数：在 utils/visualize.py 中扩展
```

---

## 🚢 部署便捷性

### 模型导出与推理

```python
# 保存的检查点包含完整状态
checkpoint = torch.load('checkpoints/best_model.pth')
# 包含:
#   - model_state_dict: 模型权重
#   - optimizer_state_dict: 优化器状态
#   - history: 训练历史记录
```

### 生产部署示例

```python
import torch
from models.cnn import get_model

# 加载模型
model = get_model(num_classes=10, input_channels=1)
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 单张推理
@torch.no_grad()
def predict(image_tensor):
    """image_tensor: [1, 1, 28, 28] 归一化后的张量"""
    output = model(image_tensor)
    predicted = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()
    return predicted, confidence
```

### Docker 部署（推荐）

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py", "--epochs", "10"]
```

```bash
# 构建并运行
docker build -t mnist-cnn .
docker run -v $(pwd)/checkpoints:/app/checkpoints mnist-cnn
```

---

## 🔧 配置参考 (config.py)

| 分类 | 变量 | 默认值 | 说明 |
|------|------|--------|------|
| **设备** | `DEVICE` | auto | CUDA/CPU 自动选择 |
| **数据** | `BATCH_SIZE` | 64 | 批次大小 |
| **数据** | `NUM_WORKERS` | 0 | 数据加载线程数 (Windows=0) |
| **数据** | `MNIST_MEAN/STD` | 0.1307/0.3081 | 标准化参数 |
| **模型** | `NUM_CLASSES` | 10 | 分类类别数 |
| **模型** | `INPUT_CHANNELS` | 1 | 输入通道数 |
| **训练** | `EPOCHS` | 10 | 训练轮数 |
| **训练** | `LEARNING_RATE` | 0.001 | 学习率 |
| **训练** | `WEIGHT_DECAY` | 1e-4 | L2 正则化系数 |
| **路径** | `CHECKPOINT_DIR` | ./checkpoints | 模型保存目录 |
| **路径** | `RESULTS_DIR` | ./results | 结果输出目录 |

> 💡 **提示**: 修改 `config.py` 即可全局调整参数，无需改动业务代码。

---

## 📈 可选依赖

```bash
# 用于混淆矩阵可视化（已内置优雅降级）
pip install scikit-learn seaborn
```

---

## 📄 License

MIT License
