#!/usr/bin/env python3
"""
验证提取的数据是否正确 - 适配前20帧→第25帧设置，256x256版本
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import json

def verify_data():
    """验证数据完整性"""
    print("开始验证数据...")
    print("帧设置: 输入前20帧 → 预测第25帧（跳过4帧）")
    print("分辨率: 256x256")
    
    # 加载元数据
    metadata_dir = Path("processed_data_256/metadata")
    
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
                expected_shape = (20, 256, 256, 3)  # 更新期望形状为256x256
                actual_shape = input_frames.shape
                print(f"    ✅ 输入帧: {actual_shape} (期望: {expected_shape})")
                
                if actual_shape != expected_shape:
                    print(f"    ⚠️  输入帧形状不匹配!")
                
                # 检查目标帧
                target_frame = np.load(sample['target_frame_path'])
                expected_target_shape = (256, 256, 3)  # 更新期望形状为256x256
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
                else:
                    print(f"    ✅ 输入帧数据范围正常 (0-255)")
                    
                if target_frame.max() <= 1.0:
                    print(f"    ⚠️  目标帧可能已经归一化 (最大值: {target_frame.max():.3f})")
                else:
                    print(f"    ✅ 目标帧数据范围正常 (0-255)")
                
                # 检查图像质量（可选）
                if input_frames.shape[1] == 256 and input_frames.shape[2] == 256:
                    # 计算图像锐度（拉普拉斯方差）
                    laplacian_var = cv2.Laplacian(input_frames[0].astype(np.uint8), cv2.CV_64F).var()
                    print(f"    📊 第一帧锐度: {laplacian_var:.1f}")
                    
                    # 调整锐度阈值以适应256x256分辨率
                    if laplacian_var < 50:
                        print(f"    ⚠️  图像可能模糊 (锐度: {laplacian_var:.1f})")
                    else:
                        print(f"    ✅ 图像清晰度良好")
                
            except Exception as e:
                print(f"    ❌ 加载失败: {e}")
    
    # 检查示例图像
    print(f"\n=== 示例图像检查 ===")
    samples_dir = Path("processed_data_256/samples")
    for category in categories:
        category_dir = samples_dir / category
        if category_dir.exists():
            image_files = list(category_dir.glob("*.jpg"))
            print(f"  {category}: {len(image_files)} 个示例图像")
            for img_file in image_files[:2]:  # 显示前2个文件
                # 检查示例图像的分辨率
                img = cv2.imread(str(img_file))
                if img is not None:
                    print(f"    - {img_file.name} ({img.shape[1]}x{img.shape[0]})")
                else:
                    print(f"    - {img_file.name} (加载失败)")
        else:
            print(f"  {category}: 示例目录不存在")
    
    # 检查数据集信息文件
    print(f"\n=== 数据集信息检查 ===")
    info_file = metadata_dir / "dataset_info.json"
    if info_file.exists():
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
        resolution = frame_info.get('resolution', 'N/A')
        print(f"分辨率: {resolution}")
        
        # 验证分辨率信息
        if resolution != "256x256":
            print(f"⚠️  分辨率不匹配: 期望256x256，实际{resolution}")
        else:
            print(f"✅ 分辨率验证通过")
    else:
        print("❌ 数据集信息文件不存在")
    
    # 检查样本中的分辨率字段
    print(f"\n=== 样本分辨率检查 ===")
    resolution_counts = all_samples['resolution'].value_counts()
    for res, count in resolution_counts.items():
        print(f"分辨率 {res}: {count} 个样本")
        if res != "256x256":
            print(f"⚠️  发现不匹配的分辨率: {res}")
    
    # 文件大小统计
    print(f"\n=== 文件大小统计 ===")
    total_input_size = 0
    total_target_size = 0
    sample_count = min(10, len(all_samples))  # 检查前10个样本的文件大小
    
    for i, sample in all_samples.head(sample_count).iterrows():
        try:
            input_path = Path(sample['input_frames_path'])
            target_path = Path(sample['target_frame_path'])
            
            if input_path.exists():
                input_size = input_path.stat().st_size / (1024 * 1024)  # MB
                total_input_size += input_size
                print(f"  {input_path.name}: {input_size:.2f} MB")
                
            if target_path.exists():
                target_size = target_path.stat().st_size / (1024 * 1024)  # MB
                total_target_size += target_size
                print(f"  {target_path.name}: {target_size:.2f} MB")
                
        except Exception as e:
            print(f"  检查文件大小失败: {e}")
    
    if sample_count > 0:
        avg_input_size = total_input_size / sample_count
        avg_target_size = total_target_size / sample_count
        print(f"\n平均文件大小:")
        print(f"  输入帧: {avg_input_size:.2f} MB")
        print(f"  目标帧: {avg_target_size:.2f} MB")
        
        # 估算总存储需求
        total_samples = len(all_samples)
        estimated_total_size = (avg_input_size + avg_target_size) * total_samples
        print(f"估算总存储需求: {estimated_total_size:.2f} MB ({estimated_total_size/1024:.2f} GB)")
        
        # 与512x512版本的比较
        print(f"\n💡 存储效率对比:")
        print(f"  相比512x512版本，256x256版本存储需求降低约75%")
    
    # 性能预估
    print(f"\n=== 性能预估 ===")
    print(f"📊 256x256分辨率优势:")
    print(f"  ✅ 训练速度: 比512x512快约4倍")
    print(f"  ✅ 显存需求: 比512x512减少约75%")
    print(f"  ✅ 批次大小: 可支持更大的批次")
    print(f"  ✅ 收敛速度: 通常更快收敛")
    
    print(f"\n🎉 数据验证完成!")
    print(f"帧设置确认: 输入前20帧 → 预测第25帧（跳过4帧）")
    print(f"分辨率确认: 256x256")

if __name__ == "__main__":
    verify_data()