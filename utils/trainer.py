"""
模型训练器模块
封装模型训练和评估的完整流程
"""

import torch
import torch.nn as nn
from config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CHECKPOINT_DIR, BEST_MODEL_PATH


class Trainer:
    """模型训练器类"""
    
    def __init__(self, model, criterion=None, optimizer=None, lr=None, device=None):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            criterion: 损失函数, 默认使用交叉熵损失
            optimizer: 优化器, 默认使用Adam
            lr: 学习率, 若None则使用配置文件的值
            device: 运行设备, 若None则自动选择
        """
        self.device = device if device else DEVICE
        self.model = model.to(self.device)
        
        # 损失函数: 交叉熵损失 (分类任务标配)
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        
        # 优化器: Adam (自适应学习率优化算法)
        current_lr = lr if lr else LEARNING_RATE
        self.optimizer = optimizer if optimizer else torch.optim.Adam(
            self.model.parameters(),
            lr=current_lr,
            weight_decay=WEIGHT_DECAY
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            epoch_loss: 平均损失
            epoch_acc: 准确率(%)
        """
        self.model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # 1. 清空梯度
            self.optimizer.zero_grad()
            
            # 2. 前向传播
            output = self.model(data)
            
            # 3. 计算损失
            loss = self.criterion(output, target)
            
            # 4. 反向传播
            loss.backward()
            
            # 5. 更新参数
            self.optimizer.step()
            
            # 统计信息
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """
        评估模型性能
        
        Args:
            val_loader: 验证/测试数据加载器
            
        Returns:
            epoch_loss: 平均损失
            epoch_acc: 准确率(%)
        """
        self.model.eval()  # 设置为评估模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 统计信息
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader=None, epochs=None, save_best=True):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器 (可选)
            epochs: 训练轮数, 若None则使用配置文件中的值
            save_best: 是否保存最佳模型
            
        Returns:
            history: 训练历史记录字典
        """
        num_epochs = epochs if epochs else EPOCHS
        best_accuracy = 0.0
        
        print(f"\n开始训练, 共 {num_epochs} 个epoch")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            
            # 验证阶段
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% || "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
                
                # 保存最佳模型
                if save_best and val_acc > best_accuracy:
                    best_accuracy = val_acc
                    self.save_model(BEST_MODEL_PATH)
                    print(f"  >> 保存最佳模型! 准确率: {best_accuracy:.2f}%")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        
        print("-" * 60)
        print(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")
        
        return self.history
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"已加载模型: {path}")


if __name__ == "__main__":
    from models.cnn import get_model
    from .dataset import get_data_loaders
    
    # 测试训练器
    model = get_model()
    trainer = Trainer(model)
    
    train_loader, test_loader = get_data_loaders(batch_size=64)
    history = trainer.train(train_loader, test_loader, epochs=2)
