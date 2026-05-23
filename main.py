"""
主程序入口
整合数据加载、模型构建、训练和评估的完整流程
"""

import sys
import argparse
from config import print_config, BEST_MODEL_PATH, NUM_CLASSES, INPUT_CHANNELS
from models.cnn import get_model
from utils.dataset import get_data_loaders
from utils.trainer import Trainer
from utils.visualize import (
    plot_training_history,
    visualize_predictions,
    plot_confusion_matrix
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手写数字识别系统 - MNIST分类任务')
    
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小 (默认使用config.py中的配置)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (默认使用config.py中的配置)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认使用config.py中的配置)')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存最佳模型')
    parser.add_argument('--no-vis', action='store_true',
                        help='不显示可视化结果')
    parser.add_argument('--load-model', type=str, default=None,
                        help='加载已有模型路径')
    
    return parser.parse_args()


def main(args):
    """主函数"""
    # 打印配置信息
    print_config()
    
    # Step 1: 加载数据
    print("\n[Step 1] 加载MNIST数据集...")
    print("-" * 40)
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    
    # Step 2: 构建模型
    print("\n[Step 2] 构建CNN模型...")
    print("-" * 40)
    model = get_model(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)
    print(f"模型结构:\n{model}\n")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # Step 3: 创建训练器
    print("\n[Step 3] 初始化训练器...")
    print("-" * 40)
    trainer = Trainer(model, lr=args.lr)
    
    # 加载预训练模型 (如果指定)
    if args.load_model:
        print(f"加载预训练模型: {args.load_model}")
        trainer.load_model(args.load_model)
        return  # 如果只是评估,则直接返回
    
    # Step 4: 训练模型
    print("\n[Step 4] 开始训练...")
    print("-" * 40)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        save_best=not args.no_save
    )
    
    # Step 5: 可视化结果
    if not args.no_vis:
        print("\n[Step 5] 可视化训练过程...")
        print("-" * 40)
        
        # 5.1 绘制训练曲线
        plot_training_history(history, show_plot=True)
        
        # 5.2 展示预测示例
        print("\n展示预测结果示例...")
        visualize_predictions(model, test_loader, num_images=16, show_plot=True)
        
        # 5.3 绘制混淆矩阵 (可选,需要scikit-learn和seaborn)
        try:
            print("\n生成混淆矩阵...")
            plot_confusion_matrix(model, test_loader, show_plot=True)
        except ImportError:
            print("  [跳过] 需要安装 scikit-learn 和 seaborn 才能绘制混淆矩阵")
            print("  安装命令: pip install scikit-learn seaborn")
    
    # Step 6: 最终评估
    print("\n[Step 6] 最终评估报告")
    print("=" * 40)
    
    # 使用最佳模型重新评估 (如果有验证集)
    if not args.no_save:
        print("加载最佳模型进行最终评估...")
        trainer.load_model(BEST_MODEL_PATH)
    
    final_loss, final_accuracy = trainer.evaluate(test_loader)
    print(f"\n最终测试集表现:")
    print(f"  - 测试损失: {final_loss:.4f}")
    print(f"  - 测试准确率: {final_accuracy:.2f}%")
    
    # 检查是否达到目标
    target_accuracy = 90.0
    if final_accuracy >= target_accuracy:
        print(f"\n✓ 成功达到目标准确率 ({target_accuracy}%)!")
    else:
        print(f"\n✗ 未达到目标准确率 ({target_accuracy}%), 请尝试调优超参数")
    
    print("\n" + "=" * 50)
    print("程序执行完成!")
    print("=" * 50)
    
    return history, final_accuracy


if __name__ == "__main__":
    args = parse_args()
    main(args)
