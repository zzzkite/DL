#!/usr/bin/env python3
"""
PyTorch数据加载器 - 用于帧预测任务
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms

class FramePredictionDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        """
        帧预测数据集类
        
        Args:
            metadata_file: 元数据CSV文件路径
            transform: 数据变换
        """
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        
        # 打印数据集信息
        print(f"✅ 加载数据集: {len(self.metadata)} 个样本")
        category_counts = self.metadata['category'].value_counts()
        for category, count in category_counts.items():
            print(f"   {category}: {count} 个样本")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        
        try:
            # 加载输入帧（前20帧）
            input_frames = np.load(sample['input_frames_path'])  # (20, 96, 96, 3)
            # 加载目标帧（第21帧）
            target_frame = np.load(sample['target_frame_path'])  # (96, 96, 3)
            
            # 转换为PyTorch张量
            input_frames = torch.from_numpy(input_frames).float()
            target_frame = torch.from_numpy(target_frame).float()
            
            # 调整维度顺序: (T, H, W, C) -> (T, C, H, W)
            input_frames = input_frames.permute(0, 3, 1, 2)  # (20, 3, 96, 96)
            target_frame = target_frame.permute(2, 0, 1)     # (3, 96, 96)
            
            # 归一化到 [0, 1] (如果数据在0-255范围内)
            input_frames = input_frames / 255.0
            target_frame = target_frame / 255.0
            
            # 应用数据变换（如果有）
            if self.transform:
                input_frames = self.transform(input_frames)
                target_frame = self.transform(target_frame)
            
            return {
                'input_frames': input_frames,
                'target_frame': target_frame,
                'category': sample['category'],
                'video_id': sample['video_id'],
                'template': sample['template'],
                'label_text': sample['label']
            }
            
        except Exception as e:
            print(f"❌ 加载样本失败 {sample['video_id']}: {e}")
            # 返回一个空样本或跳过
            return None
    
    def get_category_indices(self, category):
        """获取特定类别的所有索引"""
        return self.metadata[self.metadata['category'] == category].index.tolist()

def create_data_loaders(batch_size=8, data_path="processed_data", num_workers=4):
    """
    创建训练和验证数据加载器
    
    Args:
        batch_size: 批次大小
        data_path: 处理数据的路径
        num_workers: 数据加载的进程数
    """
    
    # 数据变换（可以根据需要添加数据增强）
    transform = transforms.Compose([
        # 可以在这里添加数据增强，如随机翻转等
        # transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    # 创建数据集
    train_dataset = FramePredictionDataset(
        f"{data_path}/metadata/train_samples.csv", 
        transform=transform
    )
    val_dataset = FramePredictionDataset(
        f"{data_path}/metadata/val_samples.csv"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # 加速GPU数据传输
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ 创建数据加载器完成")
    print(f"   训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个批次")
    print(f"   验证集: {len(val_dataset)} 个样本, {len(val_loader)} 个批次")
    print(f"   批次大小: {batch_size}")
    
    return train_loader, val_loader

def test_data_loader():
    """测试数据加载器"""
    print("测试数据加载器...")
    
    train_loader, val_loader = create_data_loaders(batch_size=4)
    
    # 测试一个训练批次
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 1:  # 只测试第一个批次
            break
            
        print(f"\n=== 批次 {batch_idx} ===")
        print(f"输入帧形状: {batch['input_frames'].shape}")  # (4, 20, 3, 96, 96)
        print(f"目标帧形状: {batch['target_frame'].shape}")  # (4, 3, 96, 96)
        print(f"数据范围: [{batch['input_frames'].min():.3f}, {batch['input_frames'].max():.3f}]")
        print(f"类别: {batch['category'][:2]}")  # 显示前2个类别
        print(f"视频ID: {batch['video_id'][:2]}")  # 显示前2个ID
        
        # 检查数据是否在合理范围内
        assert 0.0 <= batch['input_frames'].min() <= batch['input_frames'].max() <= 1.0
        assert 0.0 <= batch['target_frame'].min() <= batch['target_frame'].max() <= 1.0
    
    print("\n🎉 数据加载器测试通过!")

if __name__ == "__main__":
    test_data_loader()