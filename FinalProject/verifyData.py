#!/usr/bin/env python3
"""
验证提取的数据是否正确
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2

def verify_data():
    """验证数据完整性"""
    print("开始验证数据...")
    
    # 加载元数据
    metadata_dir = Path("processed_data/metadata")
    all_samples = pd.read_csv(metadata_dir / "all_samples.csv")
    
    print(f"总样本数: {len(all_samples)}")
    
    # 检查每个类别的样本
    categories = all_samples['category'].unique()
    
    for category in categories:
        category_samples = all_samples[all_samples['category'] == category]
        print(f"\n{category}: {len(category_samples)} 个样本")
        
        # 随机检查几个样本
        for i, sample in category_samples.head(3).iterrows():
            print(f"  检查样本 {sample['video_id']}:")
            
            # 检查输入帧
            try:
                input_frames = np.load(sample['input_frames_path'])
                print(f"    ✅ 输入帧: {input_frames.shape}")
                
                # 检查目标帧
                target_frame = np.load(sample['target_frame_path'])
                print(f"    ✅ 目标帧: {target_frame.shape}")
                
                # 检查数据范围
                print(f"    ✅ 输入帧范围: [{input_frames.min():.1f}, {input_frames.max():.1f}]")
                print(f"    ✅ 目标帧范围: [{target_frame.min():.1f}, {target_frame.max():.1f}]")
                
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
    
    # 检查示例图像
    print("\n检查示例图像...")
    samples_dir = Path("processed_data/samples")
    for category in categories:
        category_dir = samples_dir / category
        if category_dir.exists():
            image_files = list(category_dir.glob("*.jpg"))
            print(f"  {category}: {len(image_files)} 个示例图像")
    
    print("\n🎉 数据验证完成!")

if __name__ == "__main__":
    verify_data()