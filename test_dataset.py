"""
数据集加载测试脚本
快速验证MNIST数据集配置是否正确

运行方式: python test_dataset.py
"""

import sys
import os


def test_data_loading():
    """完整的数据加载测试流程"""
    
    print("=" * 60)
    print("MNIST数据集加载测试")
    print("=" * 60)
    
    # ==================== 测试1: 导入检查 ====================
    print("\n[测试1] 检查模块导入...")
    try:
        from config import DATA_ROOT, BATCH_SIZE, MNIST_MEAN, MNIST_STD, NUM_WORKERS
        from utils.dataset import get_transform, get_data_loaders
        print("  ✓ 配置模块导入成功")
        print(f"     - 数据根目录: {DATA_ROOT}")
        print(f"     - 批次大小: {BATCH_SIZE}")
        print(f"     - MNIST均值: {MNIST_MEAN}, 标准差: {MNIST_STD}")
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False
    
    # ==================== 测试2: Transform检查 ====================
    print("\n[测试2] 验证数据预处理变换...")
    try:
        train_transform = get_transform(train=True)
        test_transform = get_transform(train=False)
        
        print("  ✓ 训练集Transform:")
        for t in train_transform.transforms:
            print(f"     - {t.__class__.__name__}")
        
        print("  ✓ 测试集Transform:")
        for t in test_transform.transforms:
            print(f"     - {t.__class__.__name__}")
    except Exception as e:
        print(f"  ✗ Transform创建失败: {e}")
        return False
    
    # ==================== 测试3: 数据集加载 ====================
    print("\n[测试3] 加载MNIST数据集 (首次运行会自动下载)...")
    try:
        train_loader, test_loader = get_data_loaders(batch_size=64)
        print("  ✓ 数据加载器创建成功")
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        return False
    
    # ==================== 测试4: 数据形状验证 ====================
    import torch
    
    print("\n[测试4] 验证数据张量形状...")
    try:
        train_images, train_labels = next(iter(train_loader))
        test_images, test_labels = next(iter(test_loader))
        
        print("  ✓ 训练集批次:")
        print(f"     - 图片形状: {train_images.shape}")      # 应为 [64, 1, 28, 28]
        print(f"     - 标签形状: {train_labels.shape}")       # 应为 [64]
        print(f"     - 数据类型: {train_images.dtype}")       # 应为 torch.float32
        print(f"     - 标签类型: {train_labels.dtype}")       # 应为 torch.int64
        
        print("  ✓ 测试集批次:")
        print(f"     - 图片形状: {test_images.shape}")
        print(f"     - 标签形状: {test_labels.shape}")
        
        # 验证维度是否正确
        assert train_images.dim() == 4, "图片应该是4D张量"
        assert train_images.shape[1] == 1, "灰度图通道数应为1"
        assert train_images.shape[2:] == (28, 28), "MNIST图片尺寸应为28x28"
        
        print("\n  ✓ 所有维度验证通过!")
        
    except Exception as e:
        print(f"  ✗ 形状验证失败: {e}")
        return False
    
    # ==================== 测试5: 像素值范围检查 ====================
    print("\n[测试5] 检查像素值归一化情况...")
    try:
        img_min = train_images.min().item()
        img_max = train_images.max().item()
        img_mean = train_images.mean().item()
        img_std = train_images.std().item()
        
        print(f"  像素值统计:")
        print(f"     - 最小值: {img_min:.4f} (预期约-0.42)")
        print(f"     - 最大值: {img_max:.4f} (预期约2.82)")
        print(f"     - 均值:   {img_mean:.4f} (预期接近0)")
        print(f"     - 标准差: {img_std:.4f} (预期接近1)")
        
        if abs(img_mean) < 1.0 and 0.5 < img_std < 2.0:
            print("  ✓ 归一化参数正确!")
        else:
            print("  ⚠ 归一化结果异常，请检查Normalize参数")
            
    except Exception as e:
        print(f"  ⚠ 无法计算统计信息: {e}")
    
    # ==================== 测试6: 标签分布检查 ====================
    print("\n[测试6] 检查标签类别分布...")
    try:
        # 统计训练集标签
        all_train_labels = []
        for _, labels in train_loader:
            all_train_labels.extend(labels.tolist())
            if len(all_train_labels) >= 1000:  # 只检查前1000个样本
                break
        
        from collections import Counter
        label_counts = Counter(all_train_labels)
        
        print(f"  前1000个样本的标签分布 (共{len(all_train_labels)}个):")
        for digit in range(10):
            count = label_counts.get(digit, 0)
            bar = '█' * (count // 5)  # 简单的条形图
            print(f"     数字{digit}: {count:4d}个 {bar}")
        
        # 检查是否有所有10个数字
        unique_labels = set(label_counts.keys())
        expected_labels = set(range(10))
        
        if unique_labels == expected_labels:
            print("  ✓ 包含全部10个数字类别!")
        else:
            missing = expected_labels - unique_labels
            print(f"  ⚠ 缺少数字类别: {missing}")
            
    except Exception as e:
        print(f"  ⚠ 标签分析失败: {e}")
    
    # ==================== 测试7: DataLoader迭代测试 ====================
    print("\n[测试7] 测试DataLoader完整迭代...")
    try:
        total_batches = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            total_batches += 1
            total_samples += images.size(0)
            
            if batch_idx < 3:  # 只显示前3批
                print(f"     Batch {batch_idx+1}: "
                      f"images={list(images.shape)}, labels={list(labels.shape)}")
            
            if batch_idx >= 9:  # 测试前10个batch即可
                break
        
        print(f"\n  ✓ 成功迭代 {total_batches} 个批次, 共 {total_samples} 个样本")
        print(f"  ✓ 完整训练集预计有 {len(train_loader)} 个epoch/轮")
        
    except Exception as e:
        print(f"  ✗ 迭代测试失败: {e}")
        return False
    
    # ==================== 最终总结 ====================
    print("\n" + "=" * 60)
    print("✅ 所有测试通过! 数据集配置正确")
    print("=" * 60)
    print("\n📊 数据集摘要:")
    print(f"   • 训练集大小: {len(train_loader.dataset):,} 张图片")
    print(f"   • 测试集大小: {len(test_loader.dataset):,} 张图片")
    print(f"   • 每Epoch迭代次数: {len(train_loader)} 批次")
    print(f"   • 图片尺寸: 28x28 灰度图")
    print(f"   • 类别数量: 10 (数字0-9)")
    print("=" * 60)
    
    return True


def quick_test():
    """快速测试模式 (只做基础检查)"""
    import torch
    from utils.dataset import get_data_loaders
    
    print("快速测试模式...\n")
    
    train_loader, _ = get_data_loaders(batch_size=32)
    images, labels = next(iter(train_loader))
    
    print(f"✓ 数据加载成功")
    print(f"  形状: {images.shape}, 标签: {labels[:8].tolist()}")


if __name__ == "__main__":
    # 支持命令行参数选择测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        success = test_data_loading()
        sys.exit(0 if success else 1)
