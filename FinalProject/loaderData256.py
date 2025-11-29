#!/usr/bin/env python3
"""
PyTorch数据加载器 - 支持任务特定训练 (256x256版本)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms

class TaskSpecificDataset(Dataset):
    def __init__(self, metadata_file, task_name=None, transform=None):
        """
        任务特定数据集类 - 可筛选特定任务 (256x256版本)
        
        Args:
            metadata_file: 元数据CSV文件路径
            task_name: 任务名称 ('move_object', 'drop_object', 'cover_object')，None表示所有任务
            transform: 数据变换
        """
        self.metadata = pd.read_csv(metadata_file)
        
        # 如果指定了任务名称，筛选数据
        if task_name and task_name != 'all':
            original_count = len(self.metadata)
            self.metadata = self.metadata[self.metadata['category'] == task_name]
            print(f"   筛选任务 '{task_name}': {len(self.metadata)}/{original_count} 个样本")
        
        self.transform = transform
        self.task_name = task_name
        
        # 打印数据集信息
        task_display = task_name if task_name else '所有任务'
        print(f"✅ 加载数据集 ({task_display}): {len(self.metadata)} 个样本")
        print(f"   帧设置: 输入前20帧 → 预测第25帧（跳过4帧）")
        print(f"   分辨率: 256x256")
        
        # 显示类别分布
        category_counts = self.metadata['category'].value_counts()
        for category, count in category_counts.items():
            print(f"   {category}: {count} 个样本")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        
        try:
            # 加载输入帧（前20帧）
            input_frames = np.load(sample['input_frames_path'])  # (20, 256, 256, 3)
            # 加载目标帧（第25帧）
            target_frame = np.load(sample['target_frame_path'])  # (256, 256, 3)
            
            # 转换为PyTorch张量
            input_frames = torch.from_numpy(input_frames).float()
            target_frame = torch.from_numpy(target_frame).float()
            
            # 调整维度顺序: (T, H, W, C) -> (T, C, H, W)
            input_frames = input_frames.permute(0, 3, 1, 2)  # (20, 3, 256, 256)
            target_frame = target_frame.permute(2, 0, 1)     # (3, 256, 256)
            
            # 归一化到 [0, 1] (如果数据在0-255范围内)
            input_frames = input_frames / 255.0
            target_frame = target_frame / 255.0
            
            # 应用数据变换（如果有）
            if self.transform:
                input_frames = self.transform(input_frames)
                target_frame = self.transform(target_frame)
            
            return {
                'input_frames': input_frames,  # (20, 3, 256, 256)
                'target_frame': target_frame,  # (3, 256, 256)
                'category': sample['category'],
                'video_id': sample['video_id'],
                'template': sample['template'],
                'label_text': sample['label']
            }
            
        except Exception as e:
            print(f"❌ 加载样本失败 {sample['video_id']}: {e}")
            # 返回一个空样本而不是None，避免DataLoader出错
            return self._get_empty_sample()
    
    def _get_empty_sample(self):
        """返回一个空的样本，用于处理加载失败的情况"""
        return {
            'input_frames': torch.zeros((20, 3, 256, 256)),
            'target_frame': torch.zeros((3, 256, 256)),
            'category': 'error',
            'video_id': 'error',
            'template': 'error',
            'label_text': 'error'
        }
    
    def get_category_indices(self, category):
        """获取特定类别的所有索引"""
        return self.metadata[self.metadata['category'] == category].index.tolist()
    
    def get_resolution_info(self):
        """获取数据集分辨率信息"""
        if len(self.metadata) > 0:
            sample = self.metadata.iloc[0]
            try:
                input_frames = np.load(sample['input_frames_path'])
                return f"{input_frames.shape[2]}x{input_frames.shape[1]}"  # WxH
            except:
                return "Unknown"
        return "Unknown"

def create_task_specific_loaders(task_name='all', batch_size=8, data_path="processed_data_256", num_workers=4):
    """
    创建任务特定的训练、验证和测试数据加载器 (256x256版本)
    
    Args:
        task_name: 任务名称 ('move_object', 'drop_object', 'cover_object', 'all')
        batch_size: 批次大小 (256x256可以支持更大的批次)
        data_path: 处理数据的路径
        num_workers: 数据加载的进程数
    """
    
    # 数据变换 - 可以添加针对256x256的数据增强
    transform = transforms.Compose([
        # 可以在这里添加数据增强，如随机裁剪、翻转等
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(256, padding=8),  # 对于256x256，可以使用更大的padding
    ])
    
    # 创建任务特定数据集
    train_dataset = TaskSpecificDataset(
        f"{data_path}/metadata/train_samples.csv", 
        task_name=task_name,
        transform=transform
    )
    val_dataset = TaskSpecificDataset(
        f"{data_path}/metadata/val_samples.csv",
        task_name=task_name
    )
    test_dataset = TaskSpecificDataset(
        f"{data_path}/metadata/test_samples.csv",
        task_name=task_name
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    task_display = task_name if task_name != 'all' else '所有任务'
    resolution = train_dataset.get_resolution_info()
    
    print(f"✅ 创建任务 '{task_display}' 数据加载器完成")
    print(f"   训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个批次")
    print(f"   验证集: {len(val_dataset)} 个样本, {len(val_loader)} 个批次")
    print(f"   测试集: {len(test_dataset)} 个样本, {len(test_loader)} 个批次")
    print(f"   批次大小: {batch_size}")
    print(f"   分辨率: {resolution}")
    print(f"   输入帧形状: (20, 3, {resolution.split('x')[1]}, {resolution.split('x')[0]})")
    print(f"   目标帧形状: (3, {resolution.split('x')[1]}, {resolution.split('x')[0]})")
    
    return train_loader, val_loader, test_loader

# 保留原来的函数用于兼容性
def create_data_loaders(batch_size=8, data_path="processed_data_256", num_workers=4):
    """创建所有任务混合的数据加载器"""
    return create_task_specific_loaders('all', batch_size, data_path, num_workers)

def test_task_specific_loader():
    """测试任务特定数据加载器"""
    print("测试任务特定数据加载器 (256x256版本)...")
    
    tasks = ['move_object', 'drop_object', 'cover_object', 'all']
    
    for task in tasks:
        print(f"\n=== 测试任务: {task} ===")
        try:
            train_loader, val_loader, test_loader = create_task_specific_loaders(
                task_name=task, 
                batch_size=4  # 测试时使用较小的批次
            )
            
            # 测试一个训练批次
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 1:  # 只测试第一个批次
                    break
                    
                print(f"批次 {batch_idx}:")
                print(f"  输入帧形状: {batch['input_frames'].shape}")
                print(f"  目标帧形状: {batch['target_frame'].shape}")
                print(f"  类别分布: {pd.Series(batch['category']).value_counts().to_dict()}")
                
                # 检查数据范围
                print(f"  输入帧范围: [{batch['input_frames'].min():.3f}, {batch['input_frames'].max():.3f}]")
                print(f"  目标帧范围: [{batch['target_frame'].min():.3f}, {batch['target_frame'].max():.3f}]")
                
        except Exception as e:
            print(f"❌ 测试任务 {task} 失败: {e}")
    
    print("\n🎉 任务特定数据加载器测试通过!")

def estimate_memory_usage(batch_size=8):
    """估算内存使用情况 - 256x256版本"""
    print("\n=== 内存使用估算 (256x256) ===")
    
    # 输入帧: (batch_size, 20, 3, 256, 256) - float32
    input_memory = batch_size * 20 * 3 * 256 * 256 * 4 / (1024**3)  # GB
    
    # 目标帧: (batch_size, 3, 256, 256) - float32  
    target_memory = batch_size * 3 * 256 * 256 * 4 / (1024**3)  # GB
    
    print(f"批次大小: {batch_size}")
    print(f"输入帧内存: {input_memory:.2f} GB")
    print(f"目标帧内存: {target_memory:.2f} GB")
    print(f"总数据内存: {input_memory + target_memory:.2f} GB")
    print(f"建议GPU内存: > {(input_memory + target_memory) * 2:.2f} GB (包含模型和梯度)")
    
    # 与512x512版本对比
    print(f"\n💡 内存效率对比:")
    print(f"  相比512x512版本，内存需求降低75%")
    print(f"  可支持更大的批次大小 (2-4倍)")

if __name__ == "__main__":
    test_task_specific_loader()
    estimate_memory_usage(batch_size=8)