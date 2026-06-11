"""
可视化模块
负责绘制训练过程中的损失曲线和准确率曲线,以及预测结果展示
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import RESULTS_DIR


def plot_training_history(history, save_path=None, show_plot=True):
    """
    绘制训练历史曲线
    
    Args:
        history: 包含训练历史记录的字典,应包含以下键:
                 - 'train_loss': 训练损失列表
                 - 'train_accuracy': 训练准确率列表
                 - 'val_loss': 验证损失列表 (可选)
                 - 'val_accuracy': 验证准确率列表 (可选)
        save_path: 图片保存路径,若None则保存到results目录
        show_plot: 是否显示图形窗口
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # 绘制损失曲线
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    axes[1].plot(epochs_range, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history and history['val_accuracy']:
        axes[1].plot(epochs_range, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path is None:
        save_path = f"{RESULTS_DIR}/training_history.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


@torch.no_grad()
def visualize_predictions(model, test_loader, num_images=16, save_path=None, show_plot=True):
    """
    可视化模型预测结果
    
    Args:
        model: 已训练的模型
        test_loader: 测试数据加载器
        num_images: 要显示的图片数量 (必须是完全平方数,如4,9,16,25...)
        save_path: 图片保存路径,若None则保存到results目录
        show_plot: 是否显示图形窗口
    """
    from config import DEVICE, MNIST_MEAN, MNIST_STD
    
    device = next(model.parameters()).device
    model.eval()
    
    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    # 获取预测结果
    outputs = model(images)
    _, predictions = outputs.max(1)
    predictions = predictions.cpu().numpy()
    labels = labels.numpy()
    
    # 反归一化以便显示
    images = images.cpu().numpy()
    images = images * MNIST_STD + MNIST_MEAN  # 反标准化
    
    # 计算网格布局
    n_rows = int(np.sqrt(num_images))
    n_cols = int(np.ceil(num_images / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.suptitle('Model Predictions on MNIST Test Set', fontsize=14, fontweight='bold')
    
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols] if n_rows > 1 else axes[idx]
        
        if idx < num_images:
            # 显示图片
            img = images[idx].squeeze()  # 去掉通道维度 [1,28,28] -> [28,28]
            ax.imshow(img, cmap='gray')
            
            # 设置标题颜色: 正确=绿色, 错误=红色
            pred = predictions[idx]
            true_label = labels[idx]
            is_correct = pred == true_label
            color = 'green' if is_correct else 'red'
            
            ax.set_title(f'Pred: {pred} | True: {true_label}', 
                        color=color, fontsize=10, fontweight='bold')
        else:
            ax.axis('off')
        
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path is None:
        save_path = f"{RESULTS_DIR}/predictions.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测结果已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(model, test_loader, save_path=None, show_plot=True):
    """
    绘制混淆矩阵
    
    Args:
        model: 已训练的模型
        test_loader: 测试数据加载器
        save_path: 保存路径
        show_plot: 是否显示
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from config import DEVICE
    
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10),
                yticklabels=range(10),
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{RESULTS_DIR}/confusion_matrix.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return cm


if __name__ == "__main__":
    print("可视化模块测试")
    print("请在主程序中运行以查看效果")
