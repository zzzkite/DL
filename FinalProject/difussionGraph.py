#!/usr/bin/env python3
"""
ControlNet训练后推理脚本
结合本地Stable Diffusion生成图像并与原始数据对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import json
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from transformers import CLIPTokenizer, CLIPTextModel
    print("✅ 成功导入Diffusers和Transformers模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请安装: pip install diffusers transformers")
    sys.exit(1)

# ==================== 时序特征提取器（与训练时相同） ====================

class EnhancedTemporalFeatureExtractor(nn.Module):
    """增强版时序特征提取器 - 专门处理20帧→25帧预测"""
    
    def __init__(self, input_channels=3, feature_channels=32):
        super().__init__()
        
        # 运动轨迹预测网络
        self.trajectory_predictor = nn.Sequential(
            nn.Conv3d(input_channels, 16, (5, 3, 3), padding=(2, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, None, None)),
            nn.Conv3d(16, 8, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(8, 4, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU()
        )
        
        # 光流特征提取
        self.flow_processor = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def compute_optical_flow_sequence(self, frames):
        """计算连续光流序列"""
        batch_size, num_frames, C, H, W = frames.shape
        flow_sequences = []
        
        for b in range(batch_size):
            frame_flows = []
            for t in range(num_frames - 1):
                prev_frame = (frames[b, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                curr_frame = (frames[b, t+1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
                
                # 计算光流
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # 转换为张量
                flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()
                frame_flows.append(flow_tensor)
            
            if frame_flows:
                flow_sequence = torch.stack(frame_flows)
            else:
                flow_sequence = torch.zeros(num_frames-1, 2, H, W)
            flow_sequences.append(flow_sequence)
        
        return torch.stack(flow_sequences).to(frames.device)
    
    def predict_future_motion(self, flows):
        """预测未来运动趋势"""
        batch_size, seq_len, C, H, W = flows.shape
        
        # 使用最近几帧的运动来预测未来趋势
        recent_flows = flows[:, -4:]
        
        # 计算运动加速度
        if recent_flows.shape[1] >= 3:
            flow_acceleration = recent_flows[:, -1] - recent_flows[:, -2]
        else:
            flow_acceleration = torch.zeros_like(recent_flows[:, -1])
        
        # 预测未来运动
        last_flow = recent_flows[:, -1]
        predicted_flow = last_flow + flow_acceleration * 1.2
        
        return predicted_flow
    
    def forward(self, frames):
        """提取时序特征"""
        batch_size, num_frames, C, H, W = frames.shape
        
        # 1. 计算光流序列
        flow_sequence = self.compute_optical_flow_sequence(frames)
        
        # 2. 预测未来运动
        predicted_flow = self.predict_future_motion(flow_sequence)
        
        # 3. 处理光流特征
        flow_features = self.flow_processor(predicted_flow)
        
        # 4. 轨迹特征提取
        trajectory_input = frames.permute(0, 2, 1, 3, 4)
        trajectory_features = self.trajectory_predictor(trajectory_input)
        trajectory_features = trajectory_features.mean(dim=2)
        
        # 5. 特征融合
        combined_features = torch.cat([flow_features, trajectory_features], dim=1)
        enhanced_control = self.feature_fusion(combined_features)
        
        return enhanced_control

# ==================== 数据加载器 ====================

class FramePredictionDataset:
    """简化版数据加载器，用于加载测试数据"""
    
    def __init__(self, metadata_file, task_name=None):
        self.metadata = pd.read_csv(metadata_file)
        
        if task_name and task_name != 'all':
            self.metadata = self.metadata[self.metadata['category'] == task_name]
        
        print(f"✅ 加载推理数据集: {len(self.metadata)} 个样本")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        
        try:
            # 加载输入帧（前20帧）
            input_frames = np.load(sample['input_frames_path'])
            # 加载目标帧（第25帧）
            target_frame = np.load(sample['target_frame_path'])
            
            # 转换为PyTorch张量
            input_frames = torch.from_numpy(input_frames).float()
            target_frame = torch.from_numpy(target_frame).float()
            
            # 调整维度顺序
            input_frames = input_frames.permute(0, 3, 1, 2)
            target_frame = target_frame.permute(2, 0, 1)
            
            # 归一化
            input_frames = input_frames / 255.0
            target_frame = target_frame / 255.0
            
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
            return None

# ==================== ControlNet推理器 ====================

class ControlNetSDInference:
    """ControlNet + Stable Diffusion 推理器"""
    
    def __init__(self, sd_model_path, controlnet_model_path, task_name, device="cuda"):
        self.device = device
        self.task_name = task_name
        
        print(f"🎯 初始化 {task_name} 任务推理管道...")
        
        # 1. 加载训练好的ControlNet
        print("📦 加载训练好的ControlNet...")
        checkpoint = torch.load(controlnet_model_path, map_location='cpu')
        
        # 2. 加载本地Stable Diffusion 1.5
        print("📦 加载本地Stable Diffusion 1.5...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
        
        # 3. 创建并加载ControlNet权重
        controlnet = ControlNetModel.from_unet(self.pipe.unet)
        controlnet.load_state_dict(checkpoint['controlnet_state_dict'])
        
        # 4. 将ControlNet添加到管道
        self.pipe.controlnet = controlnet
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
        
        # 5. 获取训练参数
        self.conditioning_scale = checkpoint.get('current_scale', 1.0)
        self.training_config = checkpoint.get('config', {})
        
        # 6. 加载时序特征提取器
        self.temporal_extractor = EnhancedTemporalFeatureExtractor().to(device)
        self.temporal_extractor.eval()
        
        print(f"✅ {task_name} 推理管道初始化完成")
        print(f"   条件缩放系数: {self.conditioning_scale}")
        print(f"   训练配置: {self.training_config.get('conditioning_strategy', 'unknown')}")
    
    def prepare_control_image(self, input_frames):
        """准备控制图像（与训练时相同）"""
        with torch.no_grad():
            if len(input_frames.shape) == 4:  # (B, T, C, H, W)
                control_images = self.temporal_extractor(input_frames)
            else:  # (T, C, H, W)
                control_images = self.temporal_extractor(input_frames.unsqueeze(0))
                control_images = control_images.squeeze(0)
            
            control_images = torch.clamp(control_images, -1.0, 1.0)
            control_images = (control_images + 1.0) / 2.0
            return control_images
    
    def predict_frame(self, input_frames, text_prompt, num_inference_steps=20, guidance_scale=7.5):
        """
        预测第25帧
        
        Args:
            input_frames: (20, 3, H, W) 前20帧序列
            text_prompt: 文本描述
        """
        # 准备控制图像
        input_batch = input_frames.unsqueeze(0).to(self.device)  # (1, 20, 3, H, W)
        control_image = self.prepare_control_image(input_batch)  # (1, 3, H, W)
        
        # 调整控制图像尺寸以匹配Stable Diffusion
        control_image_pil = self.tensor_to_pil(control_image.squeeze(0))
        
        # 使用ControlNet生成图像
        with torch.no_grad():
            result = self.pipe(
                prompt=text_prompt,
                image=control_image_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=self.conditioning_scale,
                generator=torch.manual_seed(42),
                height=96,
                width=96
            )
        
        return result.images[0]
    
    def tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        tensor = tensor.cpu().squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).numpy().astype(np.uint8)
        return Image.fromarray(tensor)
    
    def pil_to_tensor(self, pil_image):
        """将PIL图像转换为张量"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).permute(2, 0, 1)
        return tensor

# ==================== 对比评估系统 ====================

class ComparisonEvaluator:
    """对比评估系统"""
    
    def __init__(self, output_dir="inference_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def compute_metrics(self, predicted, ground_truth):
        """计算图像质量指标"""
        pred_np = np.array(predicted)
        gt_np = np.array(ground_truth)
        
        # 确保图像尺寸相同
        if pred_np.shape != gt_np.shape:
            pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]))
        
        # 计算指标
        ssim_value = ssim(gt_np, pred_np, multichannel=True, channel_axis=2)
        psnr_value = psnr(gt_np, pred_np)
        
        return {
            'ssim': ssim_value,
            'psnr': psnr_value
        }
    
    def create_comparison_image(self, input_frames, predicted_frame, ground_truth, metrics, save_path):
        """创建对比图像"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 输入帧示例（第一帧、中间帧、最后一帧）
        input_indices = [0, 10, 19]
        for i, idx in enumerate(input_indices):
            frame = input_frames[idx].permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f'输入帧 {idx+1}')
            axes[0, i].axis('off')
        
        # 预测结果
        axes[1, 0].imshow(predicted_frame)
        axes[1, 0].set_title('预测帧 (第25帧)')
        axes[1, 0].axis('off')
        
        # 真实结果
        if ground_truth is not None:
            gt_frame = ground_truth.permute(1, 2, 0).cpu().numpy()
            axes[1, 1].imshow(gt_frame)
            axes[1, 1].set_title('真实帧 (第25帧)')
            axes[1, 1].axis('off')
            
            # 差异图
            pred_np = np.array(predicted_frame)
            if pred_np.shape != gt_np.shape:
                pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]))
            
            diff = np.abs(pred_np.astype(float) - gt_np.astype(float))
            diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
            
            axes[1, 2].imshow(diff_normalized, cmap='hot')
            axes[1, 2].set_title('差异图')
            axes[1, 2].axis('off')
            
            # 添加指标文本
            metrics_text = f"SSIM: {metrics['ssim']:.4f}\nPSNR: {metrics['psnr']:.2f} dB"
            fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        else:
            # 如果没有真实帧，只显示预测结果
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ==================== 主执行函数 ====================

def load_all_predictors(sd_model_path, tasks_config, device="cuda"):
    """加载所有任务的预测器"""
    predictors = {}
    for task_name, model_path in tasks_config.items():
        if Path(model_path).exists():
            try:
                predictors[task_name] = ControlNetSDInference(
                    sd_model_path, model_path, task_name, device
                )
                print(f"✅ 成功加载 {task_name} 预测器")
            except Exception as e:
                print(f"❌ 加载 {task_name} 预测器失败: {e}")
        else:
            print(f"⚠️  找不到 {task_name} 的模型文件: {model_path}")
    
    return predictors

def single_sample_inference(predictors, task_name, input_frames, text_prompt, output_dir="single_results"):
    """单样本推理"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if task_name not in predictors:
        print(f"❌ 找不到 {task_name} 的预测器")
        return
    
    predictor = predictors[task_name]
    evaluator = ComparisonEvaluator(output_dir)
    
    print(f"🎯 开始单样本推理 ({task_name})...")
    
    # 预测帧
    predicted_frame = predictor.predict_frame(input_frames, text_prompt)
    
    # 保存预测结果
    save_path = output_dir / f"{task_name}_predicted.png"
    predicted_frame.save(save_path)
    print(f"✅ 保存预测结果: {save_path}")
    
    # 创建对比图像（不包含真实帧）
    comparison_path = output_dir / f"{task_name}_comparison.png"
    evaluator.create_comparison_image(input_frames, predicted_frame, None, {}, comparison_path)
    print(f"✅ 保存对比图像: {comparison_path}")
    
    return predicted_frame

def batch_evaluation(predictors, data_loader, task_name, output_dir="batch_results", num_samples=10):
    """批量评估"""
    output_dir = Path(output_dir) / task_name
    output_dir.mkdir(exist_ok=True)
    
    if task_name not in predictors:
        print(f"❌ 找不到 {task_name} 的预测器")
        return
    
    predictor = predictors[task_name]
    evaluator = ComparisonEvaluator(output_dir)
    
    all_metrics = []
    
    print(f"🎯 开始批量评估 {task_name} 任务...")
    
    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
        
        if batch_idx * batch['input_frames'].shape[0] >= num_samples:
            break
        
        input_frames_batch = batch['input_frames']
        target_frames_batch = batch['target_frame']
        text_descriptions = batch.get('label_text', [''] * len(input_frames_batch))
        video_ids = batch.get('video_id', [f'batch_{batch_idx}_{i}' for i in range(len(input_frames_batch))])
        
        for i in range(len(input_frames_batch)):
            if len(all_metrics) >= num_samples:
                break
            
            try:
                # 预测
                predicted_frame = predictor.predict_frame(
                    input_frames_batch[i], 
                    text_descriptions[i]
                )
                
                # 计算指标
                metrics = evaluator.compute_metrics(predicted_frame, target_frames_batch[i])
                all_metrics.append(metrics)
                
                # 保存对比图像
                save_path = output_dir / f"{video_ids[i]}_comparison.png"
                evaluator.create_comparison_image(
                    input_frames_batch[i], predicted_frame, target_frames_batch[i], 
                    metrics, save_path
                )
                
                # 单独保存预测结果
                pred_save_path = output_dir / f"{video_ids[i]}_predicted.png"
                predicted_frame.save(pred_save_path)
                
                print(f"✅ {video_ids[i]} - SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}")
                
            except Exception as e:
                print(f"❌ 评估失败 {video_ids[i]}: {e}")
    
    # 汇总统计
    if all_metrics:
        ssim_values = [m['ssim'] for m in all_metrics]
        psnr_values = [m['psnr'] for m in all_metrics]
        
        summary = {
            'task': task_name,
            'num_samples': len(all_metrics),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim_min': np.min(ssim_values),
            'ssim_max': np.max(ssim_values),
            'psnr_min': np.min(psnr_values),
            'psnr_max': np.max(psnr_values)
        }
        
        # 保存汇总结果
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 打印汇总
        print(f"\n📊 {task_name} 评估汇总:")
        print(f"   样本数量: {summary['num_samples']}")
        print(f"   SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        print(f"   PSNR: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB")
        
        return summary
    else:
        print(f"❌ {task_name} 没有成功评估的样本")
        return None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ControlNet训练后推理")
    parser.add_argument("--sd_path", type=str, required=True, 
                       help="本地Stable Diffusion模型路径")
    parser.add_argument("--data_path", type=str, default="processed_data",
                       help="处理数据路径")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="输出目录")
    parser.add_argument("--mode", type=str, default="batch", choices=["single", "batch"],
                       help="推理模式: single(单样本) 或 batch(批量)")
    parser.add_argument("--task", type=str, default="all",
                       choices=['move_object', 'drop_object', 'cover_object', 'all'],
                       help="指定任务")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="批量评估的样本数量")
    
    args = parser.parse_args()
    
    # 任务配置
    tasks_config = {
        'move_object': 'training_results_move_object/controlnet_move_object_best.pth',
        'drop_object': 'training_results_drop_object/controlnet_drop_object_best.pth',
        'cover_object': 'training_results_cover_object/controlnet_cover_object_best.pth'
    }
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 使用设备: {device}")
    
    # 加载预测器
    print("📦 初始化预测器...")
    predictors = load_all_predictors(args.sd_path, tasks_config, device)
    
    if not predictors:
        print("❌ 没有可用的预测器，请检查模型文件路径")
        return
    
    if args.mode == "single":
        # 单样本推理模式
        print("\n🎯 单样本推理模式")
        
        # 这里可以手动指定输入帧和文本提示
        # 示例：创建一个简单的测试数据
        print("⚠️  单样本模式需要手动提供输入数据")
        print("请修改代码中的 input_frames 和 text_prompt")
        
        # 示例代码：
        # input_frames = torch.rand(20, 3, 96, 96)  # 替换为真实数据
        # text_prompt = "Moving something to the right"
        # single_sample_inference(predictors, args.task, input_frames, text_prompt, args.output_dir)
        
    else:
        # 批量评估模式
        print("\n🎯 批量评估模式")
        
        if args.task == 'all':
            tasks_to_evaluate = list(predictors.keys())
        else:
            tasks_to_evaluate = [args.task]
        
        all_summaries = {}
        
        for task_name in tasks_to_evaluate:
            print(f"\n{'='*50}")
            print(f"🎯 评估任务: {task_name}")
            print(f"{'='*50}")
            
            # 加载测试数据
            test_dataset = FramePredictionDataset(
                f"{args.data_path}/metadata/test_samples.csv",
                task_name=task_name
            )
            
            # 创建简单数据加载器
            class SimpleDataLoader:
                def __init__(self, dataset, batch_size=2):
                    self.dataset = dataset
                    self.batch_size = batch_size
                
                def __iter__(self):
                    self.idx = 0
                    return self
                
                def __next__(self):
                    if self.idx >= len(self.dataset):
                        raise StopIteration
                    
                    batch = []
                    for i in range(self.batch_size):
                        if self.idx >= len(self.dataset):
                            break
                        sample = self.dataset[self.idx]
                        if sample is not None:
                            batch.append(sample)
                        self.idx += 1
                    
                    if not batch:
                        raise StopIteration
                    
                    # 合并批次
                    return {
                        'input_frames': torch.stack([item['input_frames'] for item in batch]),
                        'target_frame': torch.stack([item['target_frame'] for item in batch]),
                        'label_text': [item['label_text'] for item in batch],
                        'video_id': [item['video_id'] for item in batch]
                    }
            
            data_loader = SimpleDataLoader(test_dataset, batch_size=2)
            
            # 执行评估
            summary = batch_evaluation(
                predictors, data_loader, task_name, args.output_dir, args.num_samples
            )
            
            if summary:
                all_summaries[task_name] = summary
        
        # 生成总体报告
        if all_summaries:
            generate_final_report(all_summaries, args.output_dir)
    
    print(f"\n🎊 推理完成! 结果保存在: {args.output_dir}")

def generate_final_report(summaries, output_dir):
    """生成最终评估报告"""
    report = {
        'overall': {},
        'tasks': summaries
    }
    
    # 保存报告
    report_path = Path(output_dir) / "final_evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印报告
    print(f"\n📋 最终评估报告:")
    print(f"{'='*60}")
    for task_name, summary in summaries.items():
        print(f"{task_name}:")
        print(f"  SSIM: {summary['ssim_mean']:.4f} (±{summary['ssim_std']:.4f})")
        print(f"  PSNR: {summary['psnr_mean']:.2f} dB (±{summary['psnr_std']:.2f})")
        print(f"  样本数: {summary['num_samples']}")
        print()

if __name__ == "__main__":
    main()