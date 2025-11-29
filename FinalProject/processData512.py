#!/usr/bin/env python3
"""
Something-Something V2 数据集处理脚本 - 512x512版本
输入前20帧，预测第25帧（跳过中间4帧）
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random

class VideoProcessor:
    def __init__(self, video_dir="extracted_videos/20bn-something-something-v2", 
                 label_dir="labels",
                 output_dir="processed_data_512"):
        self.video_dir = Path(video_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.target_size = (512, 512)  # 修改为512x512
        self.setup_directories()
        
    def setup_directories(self):
        """创建输出目录结构"""
        directories = [
            "frames/move_object",
            "frames/drop_object", 
            "frames/cover_object",
            "metadata",
            "samples"
        ]
        
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✅ 目录结构创建完成 - 分辨率: {self.target_size[0]}x{self.target_size[1]}")
    
    def load_dataset_labels(self, split='train'):
        """加载数据集标签"""
        label_file = self.label_dir / f"{split}.json"
        
        # 读取并修复JSON格式
        samples = []
        with open(label_file, 'r') as f:
            content = f.read().strip()
            # 修复JSON格式
            if not content.startswith('['):
                content = '[' + content + ']'
            content = content.replace('},]', '}]')
            
            try:
                samples = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                # 如果JSON解析失败，尝试逐行读取
                samples = []
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line != '[' and line != ']':
                            if line.endswith(','):
                                line = line[:-1]
                            try:
                                sample = json.loads(line)
                                samples.append(sample)
                            except json.JSONDecodeError:
                                continue
        
        print(f"✅ 加载了 {split} 集的 {len(samples)} 个样本")
        return samples
    
    def categorize_by_template_pattern(self, samples):
        """使用模板模式匹配来分类样本"""
        categories = {
            'move_object': {
                'patterns': [
                    "Moving [something]",
                    "Pushing [something]",
                    "Pulling [something]",
                    "Moving [something] from left to right",
                    "Moving [something] from right to left",
                    "Pushing [something] from left to right",
                    "Pushing [something] from right to left",
                    "Pulling [something] from left to right", 
                    "Pulling [something] from right to left",
                    "Moving [something] up",
                    "Moving [something] down"
                ],
                'samples': []
            },
            'drop_object': {
                'patterns': [
                    "Dropping [something]",
                    "Letting [something] fall",
                    "Lifting [something] up completely, then letting it drop down",
                    "Dropping [something] onto [something]",
                    "Dropping [something] behind [something]",
                    "Dropping [something] in front of [something]",
                    "Dropping [something] into [something]",
                    "Dropping [something] next to [something]"
                ],
                'samples': []
            },
            'cover_object': {
                'patterns': [
                    "Covering [something] with [something]",
                    "Putting [something] on [something]",
                    "Putting [something] onto [something]",
                    "Putting [something] on top of [something]",
                    "Placing [something] on [something]"
                ],
                'samples': []
            }
        }
        
        # 分类样本
        for sample in samples:
            template = sample.get('template', '')
            label = sample.get('label', '').lower()
            
            matched = False
            for category, info in categories.items():
                for pattern in info['patterns']:
                    # 检查模板是否包含模式，或者标签是否包含关键词
                    if pattern.lower() in template.lower() or self.contains_keywords(label, category):
                        info['samples'].append(sample)
                        matched = True
                        break
                if matched:
                    break
        
        # 打印分类结果
        print("\n=== 样本分类结果 ===")
        for category, info in categories.items():
            print(f"{category}: {len(info['samples'])} 个样本")
            # 显示前3个样本的模板
            for sample in info['samples'][:3]:
                print(f"  - {sample['template']}")
        
        return categories
    
    def contains_keywords(self, label, category):
        """检查标签是否包含类别的关键词"""
        keywords = {
            'move_object': ['moving', 'pushing', 'pulling', 'sliding'],
            'drop_object': ['dropping', 'falling', 'letting fall'],
            'cover_object': ['covering', 'putting on', 'placing on', 'on top of']
        }
        
        return any(keyword in label for keyword in keywords.get(category, []))
    
    def select_samples(self, categorized_samples, samples_per_category=100):
        """选择样本"""
        selected_samples = {}
        
        for category, info in categorized_samples.items():
            available_samples = info['samples']
            if len(available_samples) >= samples_per_category:
                selected = random.sample(available_samples, samples_per_category)
            else:
                selected = available_samples
                print(f"⚠️  {category} 只有 {len(available_samples)} 个样本，少于要求的 {samples_per_category}")
            
            selected_samples[category] = selected
            print(f"✅ {category}: 选择了 {len(selected)} 个样本")
        
        return selected_samples
    
    def extract_frames(self, video_path, num_frames=25):
        """提取视频帧 - 提取25帧，使用前20帧作为输入，第25帧作为目标"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < num_frames:
                return None
            
            # 均匀提取25帧
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # 使用高质量插值方法上采样到512x512
                    frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LANCZOS4)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    return None
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"处理视频失败 {video_path.name}: {e}")
            return None
    
    def process_samples_with_retry(self, categorized_samples, target_per_category=200):
        """处理样本，如果失败则自动补足"""
        all_samples = []
        processed_videos = set()  # 记录已处理的视频ID
        
        # 首先检查已存在的样本
        existing_samples = self.load_existing_samples()
        for sample in existing_samples:
            all_samples.append(sample)
            processed_videos.add(sample['video_id'])
        
        print(f"已加载 {len(existing_samples)} 个现有样本")
        
        # 处理每个类别直到达到目标数量
        for category, info in categorized_samples.items():
            print(f"\n处理类别: {category}")
            
            # 计算当前已有的样本数量
            current_count = len([s for s in all_samples if s['category'] == category])
            print(f"当前已有 {current_count} 个样本，目标 {target_per_category} 个")
            
            if current_count >= target_per_category:
                print(f"✅ {category} 已达到目标数量")
                continue
            
            # 获取该类别的所有样本，排除已处理的
            available_samples = [s for s in info['samples'] if s['id'] not in processed_videos]
            
            if len(available_samples) == 0:
                print(f"❌ {category} 没有更多可用样本")
                continue
            
            # 随机打乱可用样本
            random.shuffle(available_samples)
            
            processed_count = current_count
            batch_size = min(50, len(available_samples))  # 每次处理一批
            
            for batch_start in range(0, len(available_samples), batch_size):
                if processed_count >= target_per_category:
                    break
                    
                batch = available_samples[batch_start:batch_start + batch_size]
                print(f"  处理批次 {batch_start//batch_size + 1}, 样本 {len(batch)} 个")
                
                for i, sample in enumerate(batch):
                    if processed_count >= target_per_category:
                        break
                        
                    video_id = sample['id']
                    video_path = self.video_dir / f"{video_id}.webm"
                    
                    if not video_path.exists():
                        print(f"    ⚠️  视频文件不存在: {video_path}")
                        processed_videos.add(video_id)
                        continue
                    
                    print(f"    [{processed_count+1}/{target_per_category}] 处理: {video_id}")
                    
                    # 提取25帧：前20帧作为输入，第25帧作为目标（跳过21-24帧）
                    frames = self.extract_frames(video_path, num_frames=25)
                    
                    if frames and len(frames) >= 25:
                        # 保存帧数据 - 前20帧作为输入，第25帧作为目标
                        input_frames = np.array(frames[:20])  # 前20帧作为输入
                        target_frame = np.array(frames[24])   # 第25帧作为目标（跳过中间4帧）
                        
                        input_path = self.output_dir / "frames" / category / f"{video_id}_input.npy"
                        target_path = self.output_dir / "frames" / category / f"{video_id}_target.npy"
                        
                        np.save(input_path, input_frames)
                        np.save(target_path, target_frame)
                        
                        # 保存示例图像（如果是该类别第一个成功样本）
                        if processed_count == 0 and len([s for s in all_samples if s['category'] == category]) == 0:
                            sample_dir = self.output_dir / "samples" / category
                            sample_dir.mkdir(parents=True, exist_ok=True)
                            
                            cv2.imwrite(str(sample_dir / "input_frame_0.jpg"), 
                                       cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(sample_dir / "target_frame.jpg"), 
                                       cv2.cvtColor(frames[24], cv2.COLOR_RGB2BGR))  # 第25帧作为目标
                        
                        processed_sample = {
                            'category': category,
                            'video_id': video_id,
                            'video_path': str(video_path),
                            'input_frames_path': str(input_path),
                            'target_frame_path': str(target_path),
                            'template': sample['template'],
                            'label': sample['label'],
                            'placeholders': sample.get('placeholders', []),
                            'resolution': f"{self.target_size[0]}x{self.target_size[1]}"  # 添加分辨率信息
                        }
                        
                        all_samples.append(processed_sample)
                        processed_videos.add(video_id)
                        processed_count += 1
                    else:
                        print(f"    ⚠️  无法提取帧: {video_id}")
                        processed_videos.add(video_id)
                
                # 每处理完一个批次就保存一次元数据
                self.save_metadata(all_samples)
            
            print(f"✅ {category}: 成功处理 {processed_count} 个视频")
        
        return all_samples
    
    def load_existing_samples(self):
        """加载已存在的样本"""
        existing_samples = []
        metadata_file = self.output_dir / "metadata" / "all_samples.csv"
        
        if metadata_file.exists():
            try:
                df = pd.read_csv(metadata_file)
                for _, row in df.iterrows():
                    sample = {
                        'category': row['category'],
                        'video_id': row['video_id'],
                        'video_path': row['video_path'],
                        'input_frames_path': row['input_frames_path'],
                        'target_frame_path': row['target_frame_path'],
                        'template': row.get('template', ''),
                        'label': row.get('label', ''),
                        'placeholders': eval(row.get('placeholders', '[]')) if isinstance(row.get('placeholders', ''), str) else [],
                        'resolution': row.get('resolution', '512x512')
                    }
                    existing_samples.append(sample)
                print(f"✅ 加载了 {len(existing_samples)} 个现有样本")
            except Exception as e:
                print(f"⚠️  加载现有样本失败: {e}")
        
        return existing_samples
    
    def save_metadata(self, all_samples):
        """保存元数据，包括训练集、验证集和测试集（8:1:1比例）"""
        # 确保有足够的样本进行划分
        if len(all_samples) < 10:
            print(f"⚠️  样本数量不足 {len(all_samples)}，无法划分数据集")
            return
        
        # 随机打乱所有样本
        random.shuffle(all_samples)
        
        # 按8:1:1比例划分数据集
        total_samples = len(all_samples)
        train_count = int(0.8 * total_samples)  # 80% 训练集
        val_count = int(0.1 * total_samples)    # 10% 验证集
        test_count = total_samples - train_count - val_count  # 10% 测试集
        
        train_samples = all_samples[:train_count]
        val_samples = all_samples[train_count:train_count + val_count]
        test_samples = all_samples[train_count + val_count:]
        
        # 保存数据集信息
        dataset_info = {
            "name": "Something-Something V2 Processed Dataset",
            "total_samples": total_samples,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "split_ratio": "train:val:test = {}:{}:{}".format(len(train_samples), len(val_samples), len(test_samples)),
            "frame_info": {
                "input_frames": 20,    # 前20帧作为输入
                "target_frame": 1,     # 第25帧作为目标（跳过中间4帧）
                "skip_frames": 4,      # 跳过的帧数
                "resolution": f"{self.target_size[0]}x{self.target_size[1]}"  # 更新分辨率
            }
        }
        
        with open(self.output_dir / "metadata" / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # 保存样本数据
        all_df = pd.DataFrame(all_samples)
        train_df = pd.DataFrame(train_samples)
        val_df = pd.DataFrame(val_samples)
        test_df = pd.DataFrame(test_samples)
        
        all_df.to_csv(self.output_dir / "metadata" / "all_samples.csv", index=False)
        train_df.to_csv(self.output_dir / "metadata" / "train_samples.csv", index=False)
        val_df.to_csv(self.output_dir / "metadata" / "val_samples.csv", index=False)
        test_df.to_csv(self.output_dir / "metadata" / "test_samples.csv", index=False)
        
        # 打印统计信息
        print(f"\n✅ 元数据已保存")
        print(f"   总样本数: {total_samples}")
        print(f"   训练集: {len(train_samples)} (80%)")
        print(f"   验证集: {len(val_samples)} (10%)")
        print(f"   测试集: {len(test_samples)} (10%)")
        print(f"   帧设置: 输入前20帧 → 预测第25帧（跳过4帧）")
        print(f"   分辨率: {self.target_size[0]}x{self.target_size[1]}")
        
        # 类别统计（按数据集划分）
        print("\n=== 数据集类别分布 ===")
        for dataset_name, dataset in [("训练集", train_samples), ("验证集", val_samples), ("测试集", test_samples)]:
            category_counts = {}
            for sample in dataset:
                cat = sample['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print(f"\n{dataset_name} ({len(dataset)} 个样本):")
            for cat, count in category_counts.items():
                print(f"  {cat}: {count} 个样本")
    
    def process(self):
        """主处理流程"""
        print("开始数据处理...")
        print(f"📹 帧设置: 输入前20帧 → 预测第25帧（跳过中间4帧）")
        print(f"📏 分辨率: {self.target_size[0]}x{self.target_size[1]}")
        
        # 加载训练集标签
        train_samples = self.load_dataset_labels('train')
        
        # 分类样本
        categorized_samples = self.categorize_by_template_pattern(train_samples)
        
        # 处理样本并自动补足到每个类别2500个
        all_samples = self.process_samples_with_retry(categorized_samples, 2500)
        
        if not all_samples:
            print("❌ 没有成功处理任何样本")
            return
        
        # 最终保存元数据
        self.save_metadata(all_samples)
        
        # 最终统计
        category_counts = {}
        for sample in all_samples:
            cat = sample['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"\n🎉 数据处理完成!")
        print(f"   总共处理了 {len(all_samples)} 个样本")
        print(f"   帧设置: 输入前20帧 → 预测第25帧（跳过4帧）")
        print(f"   分辨率: {self.target_size[0]}x{self.target_size[1]}")
        print(f"   各类别数量:")
        for cat, count in category_counts.items():
            print(f"     {cat}: {count} 个样本")

def main():
    processor = VideoProcessor()
    processor.process()

if __name__ == "__main__":
    main()