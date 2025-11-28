#!/usr/bin/env python3
"""
验证提取的数据是否正确 - 适配前20帧→第25帧设置
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2

def verify_data():
    """验证数据完整性"""
    print("开始验证数据...")
    print("帧设置: 输入前20帧 → 预测第25帧（跳过4帧）")
    
    # 加载元数据
    metadata_dir = Path("processed_data/metadata")
    
    # 检查所有数据集文件
    dataset_files = {
        "所有样本": "all_samples.csv",
        "训练集": "train_samples.csv", 
        "验证集": "val_samples.csv",
        "测试集": "test_samples.csv"
    }
    
    for name, filename in dataset_files.items():
        file_path = metadata_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"✅ {name}: {len(df)} 个样本")
        else:
            print(f"❌ {name}文件不存在: {file_path}")
    
    # 加载完整数据集信息
    all_samples = pd.read_csv(metadata_dir / "all_samples.csv")
    
    print(f"\n总样本数: {len(all_samples)}")
    
    # 检查数据集划分比例
    train_samples = pd.read_csv(metadata_dir / "train_samples.csv")
    val_samples = pd.read_csv(metadata_dir / "val_samples.csv")
    test_samples = pd.read_csv(metadata_dir / "test_samples.csv")
    
    total_actual = len(train_samples) + len(val_samples) + len(test_samples)
    print(f"训练集: {len(train_samples)} 个样本 ({len(train_samples)/total_actual*100:.1f}%)")
    print(f"验证集: {len(val_samples)} 个样本 ({len(val_samples)/total_actual*100:.1f}%)")
    print(f"测试集: {len(test_samples)} 个样本 ({len(test_samples)/total_actual*100:.1f}%)")
    
    # 检查每个类别的样本
    categories = all_samples['category'].unique()
    
    print(f"\n=== 详细样本检查 ===")
    for category in categories:
        category_samples = all_samples[all_samples['category'] == category]
        print(f"\n{category}: {len(category_samples)} 个样本")
        
        # 随机检查几个样本
        for i, sample in category_samples.head(2).iterrows():
            print(f"  检查样本 {sample['video_id']}:")
            
            # 检查输入帧
            try:
                input_frames = np.load(sample['input_frames_path'])
                expected_shape = (20, 96, 96, 3)  # 期望的形状
                actual_shape = input_frames.shape
                print(f"    ✅ 输入帧: {actual_shape} (期望: {expected_shape})")
                
                if actual_shape != expected_shape:
                    print(f"    ⚠️  输入帧形状不匹配!")
                
                # 检查目标帧
                target_frame = np.load(sample['target_frame_path'])
                expected_target_shape = (96, 96, 3)
                actual_target_shape = target_frame.shape
                print(f"    ✅ 目标帧: {actual_target_shape} (期望: {expected_target_shape})")
                
                if actual_target_shape != expected_target_shape:
                    print(f"    ⚠️  目标帧形状不匹配!")
                
                # 检查数据范围
                print(f"    ✅ 输入帧范围: [{input_frames.min():.1f}, {input_frames.max():.1f}]")
                print(f"    ✅ 目标帧范围: [{target_frame.min():.1f}, {target_frame.max():.1f}]")
                
                # 检查是否为有效图像数据
                if input_frames.max() <= 1.0:
                    print(f"    ⚠️  输入帧可能已经归一化 (最大值: {input_frames.max():.3f})")
                if target_frame.max() <= 1.0:
                    print(f"    ⚠️  目标帧可能已经归一化 (最大值: {target_frame.max():.3f})")
                
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
    
    # 检查示例图像
    print(f"\n=== 示例图像检查 ===")
    samples_dir = Path("processed_data/samples")
    for category in categories:
        category_dir = samples_dir / category
        if category_dir.exists():
            image_files = list(category_dir.glob("*.jpg"))
            print(f"  {category}: {len(image_files)} 个示例图像")
            for img_file in image_files[:2]:  # 显示前2个文件
                print(f"    - {img_file.name}")
        else:
            print(f"  {category}: 示例目录不存在")
    
    # 检查数据集信息文件
    print(f"\n=== 数据集信息检查 ===")
    info_file = metadata_dir / "dataset_info.json"
    if info_file.exists():
        import json
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        print(f"数据集名称: {dataset_info.get('name', 'N/A')}")
        print(f"总样本数: {dataset_info.get('total_samples', 'N/A')}")
        print(f"训练集: {dataset_info.get('train_samples', 'N/A')}")
        print(f"验证集: {dataset_info.get('val_samples', 'N/A')}")
        print(f"测试集: {dataset_info.get('test_samples', 'N/A')}")
        print(f"划分比例: {dataset_info.get('split_ratio', 'N/A')}")
        
        frame_info = dataset_info.get('frame_info', {})
        print(f"输入帧数: {frame_info.get('input_frames', 'N/A')}")
        print(f"目标帧数: {frame_info.get('target_frame', 'N/A')}")
        print(f"跳过帧数: {frame_info.get('skip_frames', 'N/A')}")
        print(f"分辨率: {frame_info.get('resolution', 'N/A')}")
    else:
        print("❌ 数据集信息文件不存在")
    
    print(f"\n🎉 数据验证完成!")
    print(f"帧设置确认: 输入前20帧 → 预测第25帧（跳过4帧）")

if __name__ == "__main__":
    verify_data()