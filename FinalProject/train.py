#!/usr/bin/env python3
"""
ControlNet 1.1多帧时序训练 - 前20帧预测第25帧版本（带动态条件缩放）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import math
from collections import deque
import gc
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from loaderData import create_task_specific_loaders
    
    # 尝试导入ControlNet 1.1的模块
    try:
        from cldm.model import create_model, load_state_dict
        CONTROLNET_AVAILABLE = True
        print("✅ 成功导入ControlNet模块")
    except ImportError as e:
        print(f"⚠️  无法导入cldm: {e}")
        CONTROLNET_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class EnhancedTemporalFeatureExtractor(nn.Module):
    """增强版时序特征提取器 - 专门处理20帧→25帧预测"""
    
    def __init__(self, input_channels=3, feature_channels=32):
        super().__init__()
        
        # 运动轨迹预测网络
        self.trajectory_predictor = nn.Sequential(
            nn.Conv3d(input_channels, 16, (5, 3, 3), padding=(2, 1, 1)),  # 时间维度卷积
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, None, None)),  # 压缩时间维度
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
            nn.Conv2d(16, 3, 3, padding=1),  # 输出3通道控制图像
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
                flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # (2, H, W)
                frame_flows.append(flow_tensor)
            
            if frame_flows:
                flow_sequence = torch.stack(frame_flows)  # (T-1, 2, H, W)
                flow_sequences.append(flow_sequence)
            else:
                # 如果没有光流，创建零张量
                flow_sequences.append(torch.zeros(num_frames-1, 2, H, W))
        
        return torch.stack(flow_sequences).to(frames.device)  # (B, T-1, 2, H, W)
    
    def predict_future_motion(self, flows):
        """预测未来运动趋势"""
        batch_size, seq_len, C, H, W = flows.shape
        
        # 使用最近几帧的运动来预测未来趋势
        recent_flows = flows[:, -4:]  # 取最后4帧光流 (B, 4, 2, H, W)
        
        # 计算运动加速度（光流的变化）
        if recent_flows.shape[1] >= 3:
            flow_acceleration = recent_flows[:, -1] - recent_flows[:, -2]  # 最新运动变化
        else:
            flow_acceleration = torch.zeros_like(recent_flows[:, -1])
        
        # 预测未来运动（简单线性外推）
        last_flow = recent_flows[:, -1]  # 最后一帧光流
        predicted_flow = last_flow + flow_acceleration * 1.2  # 乘以系数预测未来
        
        return predicted_flow  # (B, 2, H, W)
    
    def forward(self, frames):
        """
        提取时序特征 - 专门为20帧→25帧预测设计
        frames: (B, 20, 3, H, W)
        返回: (B, 3, H, W) 增强的控制图像
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # 1. 计算光流序列
        flow_sequence = self.compute_optical_flow_sequence(frames)  # (B, 19, 2, H, W)
        
        # 2. 预测未来运动（第20帧→第25帧）
        predicted_flow = self.predict_future_motion(flow_sequence)  # (B, 2, H, W)
        
        # 3. 处理光流特征
        flow_features = self.flow_processor(predicted_flow)  # (B, 8, H, W)
        
        # 4. 轨迹特征提取
        trajectory_input = frames.permute(0, 2, 1, 3, 4)  # (B, 3, 20, H, W)
        trajectory_features = self.trajectory_predictor(trajectory_input)  # (B, 4, 8, H, W)
        trajectory_features = trajectory_features.mean(dim=2)  # (B, 4, H, W) 平均时间维度
        
        # 5. 特征融合
        combined_features = torch.cat([flow_features, trajectory_features], dim=1)  # (B, 12, H, W)
        enhanced_control = self.feature_fusion(combined_features)  # (B, 3, H, W)
        
        return enhanced_control

class DynamicConditioningControlNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_name = config.get('task_name', 'unknown_task')
        
        print(f"🚀 使用设备: {self.device}")
        print(f"🎯 任务: {self.task_name} - 前20帧 → 预测第25帧")
        
        # 🎯 动态条件缩放参数
        self.conditioning_strategy = config.get('conditioning_strategy', 'adaptive')
        self.initial_scale = config.get('initial_conditioning_scale', 0.8)
        self.final_scale = config.get('final_conditioning_scale', 1.2)
        self.current_scale = self.initial_scale
        
        # 策略特定参数
        self.adaptive_threshold = config.get('adaptive_threshold', 0.15)
        self.scale_step = config.get('scale_step', 0.05)
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        print(f"🎯 动态条件缩放策略: {self.conditioning_strategy}")
        print(f"   初始缩放: {self.initial_scale} → 最终缩放: {self.final_scale}")
        
        # 梯度缩放器
        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler(device='cuda')
        else:
            self.scaler = None

        # 显存清理与监控设置
        self.cleanup_steps = config.get('cleanup_steps', 20)  # 每多少个batch执行一次显存清理
        self.mem_log_steps = config.get('mem_log_steps', 10)  # 每多少个batch打印一次显存信息
        self.enable_mem_logging = config.get('enable_mem_logging', True)
        
        # 初始化组件
        self.setup_models()
        self.setup_optimizers()
        
    def update_conditioning_scale(self, epoch, total_epochs, train_loss=None, val_loss=None):
        """改进的条件缩放更新策略"""
        if self.conditioning_strategy == 'fixed':
            self.current_scale = self.initial_scale
            
        elif self.conditioning_strategy == 'linear_increase':
            # 更平滑的线性增加
            progress = min(1.0, epoch / (total_epochs * 0.8))  # 前80%线性增加
            self.current_scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
            
        elif self.conditioning_strategy == 'adaptive_improved':
            # 改进的自适应策略
            if epoch > 5 and val_loss is not None and train_loss is not None:
                # 使用移动平均来平滑判断
                if not hasattr(self, 'val_loss_history'):
                    self.val_loss_history = []
                
                self.val_loss_history.append(val_loss)
                if len(self.val_loss_history) > 5:
                    self.val_loss_history.pop(0)
                
                avg_val_loss = np.mean(self.val_loss_history)
                
                # 更保守的条件调整
                if val_loss > train_loss * 1.2 and avg_val_loss > self.best_val_loss * 1.1:
                    # 明显过拟合时才增加条件控制
                    self.current_scale = min(self.final_scale, self.current_scale + self.scale_step * 0.5)
                    print(f"  🔼 检测到过拟合，温和增加条件缩放至: {self.current_scale:.3f}")
                elif train_loss > self.adaptive_threshold * 2:
                    # 训练非常困难时才减少条件控制
                    self.current_scale = max(self.initial_scale, self.current_scale - self.scale_step * 0.5)
                    print(f"  🔽 训练困难，温和减少条件缩放至: {self.current_scale:.3f}")
        
        # 确保在合理范围内
        self.current_scale = max(0.1, min(2.0, self.current_scale))
        
        return self.current_scale
    
    def apply_conditioning_scale(self, down_block_res_samples, mid_block_res_sample):
        """应用条件缩放到ControlNet输出"""
        if self.current_scale != 1.0:
            down_block_res_samples = [
                sample * self.current_scale for sample in down_block_res_samples
            ]
            mid_block_res_sample = mid_block_res_sample * self.current_scale
        
        return down_block_res_samples, mid_block_res_sample

    def setup_models(self):
        """内存优化的模型初始化"""
        print("📦 初始化模型（内存优化版）...")
        
        try:
            # 1. 首先清理GPU内存
            torch.cuda.empty_cache()
            
            # 2. 使用更节省内存的方式加载模型
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "stable-diffusion-v1-5/tokenizer",
                local_files_only=True
            )
            
            # 3. 文本编码器 - 使用fp16
            self.text_encoder = CLIPTextModel.from_pretrained(
                "stable-diffusion-v1-5/text_encoder", 
                local_files_only=True,
                torch_dtype=torch.float16  # 添加fp16
            )
            
            # 4. VAE - 使用fp16并且设置不缓存
            self.vae = AutoencoderKL.from_pretrained(
                "stable-diffusion-v1-5/vae",
                local_files_only=True,
                torch_dtype=torch.float16
            )
            
            # 5. UNet - 使用fp16
            self.unet = UNet2DConditionModel.from_pretrained(
                "stable-diffusion-v1-5/unet",
                local_files_only=True, 
                torch_dtype=torch.float16
            )
            
            # 6. 噪声调度器
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                "stable-diffusion-v1-5/scheduler",
                local_files_only=True
            )
            
            # 7. 加载ControlNet - 使用fp16
            self.controlnet = self.load_controlnet()
            
            # 8. 时序特征提取器 - 使用fp16
            print("初始化增强时序特征提取器...")
            self.temporal_extractor = EnhancedTemporalFeatureExtractor().to(self.device)
            
            # 9. 移动到设备并冻结参数
            self.text_encoder = self.text_encoder.to(self.device)
            self.vae = self.vae.to(self.device) 
            self.unet = self.unet.to(self.device)
            self.controlnet = self.controlnet.to(self.device)
            
            # 冻结不需要训练的组件
            self.text_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.temporal_extractor.requires_grad_(True) # 允许训练时序提取器
            
            # 只训练ControlNet
            self.controlnet.requires_grad_(True)
            
            # 10. 再次清理内存
            torch.cuda.empty_cache()
            
            trainable_params = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.controlnet.parameters())
            print(f"✅ 模型初始化完成")
            print(f"   ControlNet参数: {trainable_params:,} 可训练 / {total_params:,} 总计")
            
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            # 清理内存后重新抛出异常
            torch.cuda.empty_cache()
            raise

    
    def load_controlnet(self):
        """加载ControlNet模型"""
        try:
            if CONTROLNET_AVAILABLE:
                # 使用ControlNet 1.1官方加载
                controlnet_dir = Path("ControlNet-v1-1")
                model_path = controlnet_dir / "control_sd15_canny.pth"
                config_path = controlnet_dir / "cldm_v15.yaml"
                
                model = create_model(str(config_path)).to(self.device)
                model.load_state_dict(load_state_dict(str(model_path), location='cpu'))
                return model.control_model
            else:
                # 使用diffusers的ControlNet
                from diffusers import ControlNetModel
                controlnet = ControlNetModel.from_unet(
                    self.unet,
                    conditioning_channels=3
                )
                print("✅ 使用diffusers ControlNet")
                return controlnet
                
        except Exception as e:
            print(f"❌ 加载ControlNet失败: {e}")
            raise
    
    def setup_optimizers(self):
        """设置优化器 - 修复版本"""
        # 使用更大的学习率
        self.optimizer = optim.AdamW(
            self.controlnet.parameters(),
            lr=self.config.get('learning_rate', 5e-5),  # 增大学习率
            weight_decay=self.config.get('weight_decay', 1e-3),  # 减少权重衰减
            betas=(0.9, 0.999),
            eps=1e-8  # 添加eps防止除零
        )
        
        # 改进的学习率调度器
        if self.config.get('lr_scheduler') == 'cosine_with_warmup':
            # 带warmup的余弦退火
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,  # 重启周期
                T_mult=2,
                eta_min=self.config.get('min_learning_rate', 1e-6)
            )
        else:
            # 保持原来的调度器
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['num_epochs'],
                eta_min=self.config.get('min_learning_rate', 1e-6)
            )
        
        print("✅ 优化器设置完成")

    # ---- GPU 内存监控/工具方法 ----
    def _log_gpu_memory(self, label: str = ""):
        if self.device.type != 'cuda' or not self.enable_mem_logging:
            return
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_alloc = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[GPU MEM] {label} allocated={allocated:.1f}MB reserved={reserved:.1f}MB max_alloc={max_alloc:.1f}MB")
        except Exception:
            pass

    def _optimizer_state_cpu(self):
        """Return a CPU copy of optimizer.state_dict() to avoid storing GPU tensors when saving."""
        state = self.optimizer.state_dict()
        state_cpu = {'state': {}, 'param_groups': state.get('param_groups', [])}
        for k, v in state.get('state', {}).items():
            state_cpu['state'][k] = {}
            for kk, vv in v.items():
                try:
                    if isinstance(vv, torch.Tensor):
                        state_cpu['state'][k][kk] = vv.cpu()
                    else:
                        state_cpu['state'][k][kk] = vv
                except Exception:
                    state_cpu['state'][k][kk] = vv
        return state_cpu

    def _cleanup_temps(self, *tensors, force_gc: bool = False):
        """Delete provided tensor references and optionally run GC + empty_cache."""
        for t in tensors:
            try:
                del t
            except Exception:
                pass
        if force_gc:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
    
    def prepare_control_image(self, input_frames):
        """
        准备控制图像 - 使用增强的时序特征
        input_frames: (B, 20, 3, H, W)
        """
        try:
            # 使用增强时序特征提取器
            control_images = self.temporal_extractor(input_frames)
            
            # 确保输出在合理范围内
            control_images = torch.clamp(control_images, -1.0, 1.0)
            control_images = (control_images + 1.0) / 2.0  # 转换到 [0, 1]
            
            return control_images
            
        except Exception as e:
            print(f"⚠️ 时序特征提取失败: {e}, 使用备用方案")
            return self.prepare_backup_control_image(input_frames)
    
    def prepare_backup_control_image(self, input_frames):
        """备用控制图像生成"""
        batch_size = input_frames.shape[0]
        control_images = []
        
        for i in range(batch_size):
            # 使用最后一帧的Canny边缘
            last_frame = input_frames[i, -1]
            img_np = (last_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            median_intensity = np.median(img_gray)
            lower_threshold = int(max(0, 0.66 * median_intensity))
            upper_threshold = int(min(255, 1.33 * median_intensity))
            
            edges = cv2.Canny(img_gray, lower_threshold, upper_threshold)
            edges = np.repeat(edges[:, :, None], 3, axis=2)
            control_tensor = torch.from_numpy(edges).permute(2, 0, 1).float() / 255.0
            control_images.append(control_tensor)
        
        return torch.stack(control_images).to(self.device)
    
    def encode_images(self, images):
        """将图像编码到潜在空间"""
        images = images * 2.0 - 1.0  # 转换到 [-1, 1]
        
        with torch.no_grad():
            if self.device.type == 'cuda' and images.dtype != torch.float16:
                with torch.cuda.amp.autocast():
                    latents = self.vae.encode(images.to(torch.float16)).latent_dist.sample()
            else:
                latents = self.vae.encode(images).latent_dist.sample()
                
            latents = latents * self.vae.config.scaling_factor
            
        return latents
    
    def encode_text(self, text_list):
        """编码文本描述"""
        if not text_list or all(text == '' for text in text_list):
            text_list = ['a person interacting with an object'] * len(text_list) if text_list else ['interaction']
            
        inputs = self.tokenizer(
            text_list, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
            
        return text_embeddings
    
    def train_epoch(self, train_loader, epoch):
        """内存优化的训练epoch"""
        self.controlnet.train()
        total_loss = 0
        num_batches = 0
        
        # 手动清理内存
        torch.cuda.empty_cache()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            try:
                # 准备数据
                input_frames = batch['input_frames'].to(self.device)
                target_frames = batch['target_frame'].to(self.device)
                text_descriptions = batch.get('label_text', ['moving object'] * len(input_frames))
                
                # 使用更节省内存的混合精度
                with torch.amp.autocast('cuda', enabled=True):  # 修复的API
                    # 编码目标图像到潜在空间
                    target_latents = self.encode_images(target_frames)
                    
                    # 编码文本
                    text_embeddings = self.encode_text(text_descriptions)
                    
                    # 准备控制图像
                    control_images = self.prepare_control_image(input_frames)
                    
                    # 添加噪声
                    noise = torch.randn_like(target_latents)
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, 
                        (target_latents.shape[0],), device=self.device
                    ).long()
                    
                    noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
                    
                    # ControlNet前向传播
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=control_images,
                        return_dict=False,
                    )
                    
                    # 应用条件缩放
                    down_block_res_samples, mid_block_res_sample = self.apply_conditioning_scale(
                        down_block_res_samples, mid_block_res_sample
                    )
                    
                    # UNet前向传播
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                    
                    # 计算损失
                    loss = F.mse_loss(noise_pred, noise)
                
                # 反向传播
                self.optimizer.zero_grad()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    if self.config.get('grad_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.controlnet.parameters(), 
                            self.config['grad_clip']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.controlnet.parameters(), 
                            self.config['grad_clip']
                        )
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 定期打印显存信息和清理临时变量
                if batch_idx % self.mem_log_steps == 0:
                    self._log_gpu_memory(f"Epoch{epoch} Batch{batch_idx} post-step")

                # 删除大张量引用以便回收
                try:
                    del target_latents, text_embeddings, control_images, noise, timesteps, noisy_latents
                    del down_block_res_samples, mid_block_res_sample, noise_pred
                except Exception:
                    pass

                # 每隔 cleanup_steps 触发显存清理和 GC
                if (batch_idx + 1) % self.cleanup_steps == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

                if batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                        f"LR: {current_lr:.2e}, Scale: {self.current_scale:.3f}")
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  批次 {batch_idx} 内存不足，跳过")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader):
        """验证模型 - 只使用验证集"""
        self.controlnet.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                    
                input_frames = batch['input_frames'].to(self.device)
                target_frames = batch['target_frame'].to(self.device)
                text_descriptions = batch.get('label_text', ['moving object'] * len(input_frames))
                
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    target_latents = self.encode_images(target_frames)
                    text_embeddings = self.encode_text(text_descriptions)
                    control_images = self.prepare_control_image(input_frames)
                    
                    noise = torch.randn_like(target_latents)
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, 
                        (target_latents.shape[0],), device=self.device
                    ).long()
                    
                    noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
                    
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=control_images,
                        return_dict=False,
                    )
                    
                    # 🎯 验证时也应用相同的条件缩放
                    down_block_res_samples, mid_block_res_sample = self.apply_conditioning_scale(
                        down_block_res_samples, mid_block_res_sample
                    )
                    
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                    
                    loss = F.mse_loss(noise_pred, noise)
                
                total_loss += loss.item()
                num_batches += 1

                # 验证阶段也定期清理和打印显存
                if num_batches % self.mem_log_steps == 0:
                    self._log_gpu_memory(f"Validate Batch {num_batches}")

                try:
                    del target_latents, text_embeddings, control_images, noise, timesteps, noisy_latents
                    del down_block_res_samples, mid_block_res_sample, noise_pred
                except Exception:
                    pass

                if num_batches % self.cleanup_steps == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, train_loader, val_loader):
        """主训练循环 - 只使用训练集和验证集"""
        print("🚀 开始训练循环...")
        print("📊 仅使用训练集和验证集，测试集保留用于最终评估")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        scale_history = []
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n=== Epoch {epoch}/{self.config['num_epochs']} ===")
            
            # 🎯 在每个epoch开始时更新条件缩放
            if epoch > 1:
                self.update_conditioning_scale(
                    epoch, self.config['num_epochs'], 
                    train_losses[-1] if train_losses else None,
                    val_losses[-1] if val_losses else None
                )
            
            # 训练（训练集）
            train_loss = self.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # 验证（验证集）
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            scale_history.append(self.current_scale)
            
            # 更新学习率
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            print(f"✅ Epoch {epoch} 完成")
            print(f"   训练损失: {train_loss:.4f}")
            print(f"   验证损失: {val_loss:.4f}")
            print(f"   条件缩放: {self.current_scale:.3f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"💾 保存最佳模型，验证损失: {val_loss:.4f}")
            
            # 定期保存检查点
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, val_loss)
        
        # 绘制训练曲线和缩放历史
        self.plot_training_metrics(train_losses, val_losses, scale_history)
        print("🎉 训练完成!")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存模型检查点"""
        # 为了减少在保存时的GPU内存占用，把 optimizer state 转到 CPU
        optimizer_state_cpu = None
        try:
            optimizer_state_cpu = self._optimizer_state_cpu()
        except Exception:
            optimizer_state_cpu = self.optimizer.state_dict()

        checkpoint = {
            'epoch': epoch,
            'controlnet_state_dict': self.controlnet.state_dict(),
            'optimizer_state_dict': optimizer_state_cpu,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'task_name': self.task_name,
            'current_scale': self.current_scale,  # 🆕 保存当前缩放值
            'conditioning_strategy': self.conditioning_strategy  # 🆕 保存策略
        }
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        if is_best:
            filename = f"controlnet_{self.task_name}_best.pth"
        else:
            filename = f"controlnet_{self.task_name}_epoch_{epoch}.pth"
        
        save_path = output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"💾 保存检查点: {save_path}")
    
    def plot_training_metrics(self, train_losses, val_losses, scale_history):
        """绘制训练曲线和缩放历史"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 损失曲线
        ax1.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
        ax1.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'ControlNet Training - {self.task_name}\n20 frames → 25th frame prediction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 缩放历史
        ax2.plot(scale_history, label='Conditioning Scale', linewidth=2, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Scale Value')
        ax2.set_title('Dynamic Conditioning Scale History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path(self.config['output_dir'])
        plt.savefig(output_dir / f'training_metrics_{self.task_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 🎯 基础训练配置
    base_config = {
        # 基础训练参数
        'learning_rate': 1e-4,
        'min_learning_rate': 1e-6,
        'weight_decay': 1e-3,
        'num_epochs': 50,
        'batch_size': 2,
        'save_interval': 10,

        # 添加梯度累积
        'gradient_accumulation_steps': 4,  # 新的参数

        # 优化器参数
        'lr_scheduler': 'cosine',
        'lr_step_size': 20,
        'lr_gamma': 0.5,
        'warmup_steps': 500,                  # 新的参数
        
        # 训练策略参数
        'grad_clip': 1.0,
        
        # 🎯 动态条件缩放参数
        'conditioning_strategy': 'linear_increase',  # 'fixed', 'linear_increase', 'adaptive', 'stepwise', 'cosine'
        'initial_conditioning_scale': 0.6,
        'final_conditioning_scale': 1.0,
        'adaptive_threshold': 0.15,
        'scale_step': 0.05,
    }
    
    # 🎯 定义三个任务
    tasks = [
        {
            'name': 'move_object',
            'display_name': '移动物体',
            'config_override': {
                'learning_rate': 1e-4,
                'num_epochs': 50,
                'conditioning_strategy': 'adaptive',
                'initial_conditioning_scale': 0.2,  # 移动任务可以更宽松
                'final_conditioning_scale': 1.0,
                'gtad_clip': 0.5,
                'weight_decay':1e-2,
            }
        },
        {
            'name': 'drop_object', 
            'display_name': '掉落物体',
            'config_override': {
                'learning_rate': 1e-4,
                'num_epochs': 50,
                'conditioning_strategy': 'linear_increase',  # 掉落任务需要逐渐加强控制
                'initial_conditioning_scale': 0.8,
                'final_conditioning_scale': 1.3,
            }
        },
        {
            'name': 'cover_object',
            'display_name': '覆盖物体', 
            'config_override': {
                'learning_rate': 8e-6,
                'num_epochs': 50,
                'conditioning_strategy': 'stepwise',  # 覆盖任务分阶段控制
                'initial_conditioning_scale': 0.8,
                'final_conditioning_scale': 1.2,
            }
        }
    ]
    
    print("=" * 60)
    print("🎯 ControlNet 1.1 多任务分别训练（动态条件缩放）")
    print("📊 任务: 前20帧 → 预测第25帧")
    print("=" * 60)
    
    # 🔄 分别训练每个任务
    for task_info in tasks:
        task_name = task_info['name']
        task_display = task_info['display_name']
        
        print(f"\n{'='*50}")
        print(f"🚀 开始训练任务: {task_display} ({task_name})")
        print(f"{'='*50}")
        
        # 合并配置
        task_config = {**base_config, **task_info['config_override']}
        task_config['output_dir'] = f'training_results_{task_name}'
        task_config['task_name'] = task_name
        
        # 打印任务特定配置
        print(f"任务配置:")
        for key, value in task_config.items():
            if key in task_info['config_override']:
                print(f"  {key}: {value} 🎯")
            else:
                print(f"  {key}: {value}")
        
        # 创建输出目录
        output_dir = Path(task_config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # 🗂️ 加载任务特定数据
        print(f"\n📊 加载 {task_display} 数据...")
        try:
            train_loader, val_loader, test_loader = create_task_specific_loaders(
                task_name=task_name,
                batch_size=task_config['batch_size'],
                data_path="processed_data"
            )
            
            # 检查数据是否足够
            if len(train_loader.dataset) == 0:
                print(f"❌ 任务 {task_name} 没有训练数据，跳过")
                continue
                
            print(f"✅ 数据加载成功")
            print(f"   训练集: {len(train_loader.dataset)} 样本")
            print(f"   验证集: {len(val_loader.dataset)} 样本") 
            print(f"   测试集: {len(test_loader.dataset)} 样本 - 保留用于最终评估")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            continue
        
        # 🤖 初始化训练器
        try:
            trainer = DynamicConditioningControlNetTrainer(task_config)
        except Exception as e:
            print(f"❌ 训练器初始化失败: {e}")
            continue
        
        # 🏋️ 开始训练
        try:
            trainer.train(train_loader, val_loader)
            print(f"🎉 任务 {task_display} 训练完成!")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ 任务 {task_display} 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 任务 {task_display} 训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🎊 所有任务训练完成!")
    print("📁 每个任务的模型保存在各自的目录中:")
    for task_info in tasks:
        print(f"   {task_info['display_name']}: training_results_{task_info['name']}/")
    print(f"{'='*60}")

# 可选：保留原来的main函数用于单个任务训练
def main_single_task():
    """单个任务训练（用于调试或特定任务）"""
    config = {
        'learning_rate': 1e-5,
        'num_epochs': 50,
        'batch_size': 2,
        'save_interval': 10,
        'output_dir': 'training_results_single',
        'task_name': 'drop_object',  # 指定单个任务
        
        # 动态条件缩放参数
        'conditioning_strategy': 'linear_increase',
        'initial_conditioning_scale': 0.8,
        'final_conditioning_scale': 1.3,
        'adaptive_threshold': 0.15,
        'scale_step': 0.05,

        'min_learning_rate': 1e-6,
        'weight_decay': 1e-3,

        # 添加梯度累积
        'gradient_accumulation_steps': 4,  # 新的参数

        # 优化器参数
        'lr_scheduler': 'cosine',
        'lr_step_size': 20,
        'lr_gamma': 0.5,
        'warmup_steps': 500,                  # 新的参数
        
        # 训练策略参数
        'grad_clip': 1.0,
    }
    
    train_loader, val_loader, _ = create_task_specific_loaders(
        task_name=config['task_name'],
        batch_size=config['batch_size']
    )
    
    trainer = DynamicConditioningControlNetTrainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    #main()  # 使用多任务训练
    main_single_task()  # 或者使用单任务训练