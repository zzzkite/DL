#!/usr/bin/env python3
"""
ControlNet 1.1 256x256训练脚本 - 优化版本
针对训练效果不佳的立即改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.optim as optim
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import gc
import time
from typing import Dict, List, Optional, Tuple

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from loaderData256 import create_task_specific_loaders  # 修改为256数据加载器
    
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

class ControlNet256Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_name = config.get('task_name', 'unknown_task')
        
        print(f"🚀 使用设备: {self.device}")
        print(f"🎯 任务: {self.task_name}")
        print(f"📏 分辨率: 256x256")
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = config.get('patience', 12)
        
        # 梯度缩放器
        self.scaler = torch.amp.GradScaler('cuda')
        
        # 初始化组件
        self.setup_models()
        self.setup_optimizers()
        
    def setup_models(self):
        """模型初始化"""
        print("📦 初始化模型...")
        
        # 1. 加载组件
        self.tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/tokenizer", local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained("stable-diffusion-v1-5/text_encoder", local_files_only=True).to(self.device)
        self.vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/vae", local_files_only=True).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/unet", local_files_only=True).to(self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained("stable-diffusion-v1-5/scheduler", local_files_only=True)
        
        # 2. 加载 ControlNet
        self.controlnet = self.load_controlnet().to(self.device)
        
        # 3. 冻结参数
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(True)
        
        # 打印参数量
        trainable_params = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.controlnet.parameters())
        print(f"✅ 模型初始化完成，可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
        
        # 检查VAE输入尺寸兼容性
        vae_sample_size = self.vae.config.sample_size
        print(f"🔍 VAE样本尺寸: {vae_sample_size}")
        if vae_sample_size != 256 and vae_sample_size != 512:
            print(f"⚠️  VAE预期输入尺寸为{vae_sample_size}x{vae_sample_size}，数据为256x256")

    def load_controlnet(self):
        """加载ControlNet模型"""
        try:
            if CONTROLNET_AVAILABLE:
                controlnet_dir = Path("ControlNet-v1-1")
                model_path = controlnet_dir / "control_sd15_canny.pth"
                config_path = controlnet_dir / "cldm_v15.yaml"
                
                if model_path.exists():
                    print(f"📂 加载预训练权重: {model_path}")
                    model = create_model(str(config_path)).to(self.device)
                    model.load_state_dict(load_state_dict(str(model_path), location='cpu'))
                    return model.control_model
                else:
                    print("⚠️ 未找到预训练权重，将尝试从UNet初始化")
            
            # 备选方案：从 UNet 初始化
            from diffusers import ControlNetModel
            print("🆕 从 UNet 复制权重初始化 ControlNet")
            controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=3)
            return controlnet
                
        except Exception as e:
            print(f"❌ 加载ControlNet失败: {e}")
            raise

    def setup_optimizers(self):
        """优化器配置 - 立即改进：添加warmup + 更大学习率"""
        # 使用更大的学习率和更小的权重衰减
        base_lr = self.config.get('learning_rate', 5e-5)  # 提高学习率
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.controlnet.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.get('weight_decay', 1e-4),  # 减小权重衰减
                "lr": base_lr,
            },
            {
                "params": [p for n, p in self.controlnet.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": base_lr,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,  # 统一的权重衰减
            eps=1e-8
        )
        
        # 使用带warmup的余弦退火调度器
        self.lr_scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.get('warmup_steps', 500),
            num_training_steps=self.config.get('num_epochs', 40) * 100,  # 估计的步数
            num_cycles=0.5
        )

    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        """创建带warmup的余弦退火调度器"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def prepare_images_for_vae(self, images):
        """准备图像以适应VAE输入"""
        # 确保图像在[-1, 1]范围内
        if torch.max(images) <= 1.0 and torch.min(images) >= 0.0:
            images = images * 2.0 - 1.0
        
        # 如果VAE需要特定尺寸，进行调整
        vae_sample_size = self.vae.config.sample_size
        current_size = images.shape[-1]
        
        if current_size != vae_sample_size:
            print(f"🔧 调整图像尺寸从 {current_size}x{current_size} 到 {vae_sample_size}x{vae_sample_size}")
            images = F.interpolate(images, size=(vae_sample_size, vae_sample_size), 
                                 mode='bilinear', align_corners=False)
        
        return images

    def get_canny_edges(self, image_tensor, training=False):
        """
        Canny边缘检测 - 立即改进：更好的参数和数据增强
        """
        batch_size = image_tensor.shape[0]
        
        # 确保输入在 [0, 1] 范围内
        if torch.max(image_tensor) > 1.0:
            image_tensor = (image_tensor + 1.0) / 2.0
        
        # 训练时添加数据增强
        if training:
            # 随机调整亮度和对比度
            brightness = 0.1 * torch.randn(batch_size, 1, 1, 1, device=self.device)
            contrast = 1.0 + 0.2 * torch.randn(batch_size, 1, 1, 1, device=self.device)
            image_tensor = image_tensor * contrast + brightness
            image_tensor = torch.clamp(image_tensor, 0, 1)
        
        images_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        edges_list = []
        for i in range(batch_size):
            img_gray = cv2.cvtColor(images_np[i], cv2.COLOR_RGB2GRAY)
            
            # 立即改进：更宽的阈值范围，更好的边缘检测
            v = np.median(img_gray)
            sigma = 0.33
            # 使用更宽的阈值范围，确保捕捉到足够多的边缘
            lower = int(max(0, (1.0 - 2 * sigma) * v))  # 更低的阈值
            upper = int(min(255, (1.0 + 2 * sigma) * v))  # 更高的阈值
            
            edge = cv2.Canny(img_gray, lower, upper)
            
            # 立即改进：更好的形态学操作，改善边缘连续性
            kernel = np.ones((2, 2), np.uint8)  # 稍大的核
            edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
            
            # 扩展回3通道
            edge = np.stack([edge] * 3, axis=-1)
            edges_list.append(edge)
            
        edges_np = np.stack(edges_list)
        edges_tensor = torch.from_numpy(edges_np).float() / 255.0
        return edges_tensor.permute(0, 3, 1, 2).to(self.device)

    def compute_loss(self, batch, training=True):
        """统一的损失计算函数 - 适配256x256"""
        # 1. 准备数据
        current_frame_20 = batch['input_frames'][:, -1].to(self.device) 
        target_frame_25 = batch['target_frame'].to(self.device)
        text_descriptions = batch.get('label_text', ['interaction'] * len(current_frame_20))
        
        # 2. VAE编码目标图 - 适配尺寸
        target_frame_prepared = self.prepare_images_for_vae(target_frame_25)
        target_latents = self.vae.encode(target_frame_prepared).latent_dist.sample()
        target_latents = target_latents * self.vae.config.scaling_factor
        
        # 3. CLIP编码文本
        inputs = self.tokenizer(
            text_descriptions, 
            max_length=77, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]
        
        # 4. 准备ControlNet条件
        control_cond = self.get_canny_edges(current_frame_20, training=training)
        
        # 5. 采样timestep和噪声 - 使用固定的全范围
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps, 
            (target_latents.shape[0],), 
            device=self.device
        ).long()
        
        noise = torch.randn_like(target_latents)
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        
        # 6. ControlNet前向
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_cond,
            return_dict=False,
        )
        
        # 7. UNet预测
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        
        # 8. 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

    def train_epoch(self, train_loader, epoch):
        """训练epoch - 立即改进：更频繁的日志和动态监控"""
        self.controlnet.train()
        total_loss = 0
        num_batches = 0
        
        accumulation_steps = self.config.get('accumulation_steps', 2)
        
        print(f"📚 开始第 {epoch} 轮训练，共有 {len(train_loader)} 个批次")
        
        # 跟踪批次损失用于动态调整
        batch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None: 
                continue
                
            # 重置梯度
            self.optimizer.zero_grad()
            
            try:
                # 使用自动混合精度
                with torch.amp.autocast('cuda'):
                    loss = self.compute_loss(batch, training=True)
                
                # 梯度累积
                loss = loss / accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), max_norm=1.0)
                    
                    # 优化器步进
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # 更新学习率调度器
                    if hasattr(self, 'lr_scheduler'):
                        self.lr_scheduler.step()
                
                # 记录损失
                loss_value = loss.item() * accumulation_steps
                total_loss += loss_value
                num_batches += 1
                batch_losses.append(loss_value)
                
                # 立即改进：更频繁的日志输出（每5个batch）
                if batch_idx % 5 == 0:  # 从10改为5，更频繁的监控
                    current_lr = self.optimizer.param_groups[0]['lr']
                    current_step = (epoch - 1) * len(train_loader) + batch_idx
                    
                    # 计算最近几个batch的平均损失
                    recent_avg = np.mean(batch_losses[-10:]) if len(batch_losses) >= 10 else loss_value
                    
                    print(f"Epoch {epoch} | Batch {batch_idx:3d}/{len(train_loader)} | "
                          f"Loss: {loss_value:.6f} | Recent: {recent_avg:.6f} | "
                          f"LR: {current_lr:.2e} | Step: {current_step}")
                
                # 定期清理显存
                if batch_idx % 20 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    if batch_idx % 40 == 0:
                        self._log_gpu_memory(f"Epoch{epoch} Batch{batch_idx}")
                    
            except Exception as e:
                print(f"❌ 训练批次 {batch_idx} 出错: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"✅ 第 {epoch} 轮训练完成，平均损失: {avg_loss:.6f}")
        return avg_loss

    def _log_gpu_memory(self, label: str = ""):
        """GPU内存监控"""
        if self.device.type != 'cuda':
            return
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            print(f"[GPU MEM] {label} allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
        except Exception:
            pass

    def validate(self, val_loader):
        """验证函数 - 适配256x256"""
        self.controlnet.eval()
        total_loss = 0.0
        num_batches = 0
        
        print(f"🧪 开始验证，共有 {len(val_loader)} 个批次")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch is None:
                    continue
                
                try:
                    with torch.amp.autocast('cuda'):
                        loss = self.compute_loss(batch, training=False)
                    
                    loss_value = loss.item()
                    total_loss += loss_value
                    num_batches += 1
                    
                    # 更频繁的验证日志
                    if batch_idx % 3 == 0:  # 从5改为3，更频繁的验证监控
                        print(f"验证批次 {batch_idx}/{len(val_loader)} | Loss: {loss_value:.6f}")
                        
                except Exception as e:
                    print(f"❌ 验证批次 {batch_idx} 出错: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"✅ 验证完成，平均损失: {avg_loss:.6f}")
        return avg_loss

    def save_checkpoint(self, epoch, task_name, is_best=False):
        """保存检查点"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if is_best:
            save_path = output_dir / f"controlnet_{task_name}_best.pth"
        else:
            save_path = output_dir / f"controlnet_{task_name}_epoch_{epoch}.pth"
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.controlnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if hasattr(self, 'lr_scheduler') else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        print(f"💾 模型已保存: {save_path}")

    def plot_and_save_losses(self, train_losses, val_losses=None):
        """绘制训练/验证损失"""
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 6))
            epochs = list(range(1, len(train_losses) + 1))
            
            plt.plot(epochs, train_losses, marker='o', color='tab:blue', label='Train Loss', linewidth=2)
            if val_losses is not None and len(val_losses) == len(train_losses):
                plt.plot(epochs, val_losses, marker='s', color='tab:orange', label='Val Loss', linewidth=2)
                
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training / Validation Loss - {self.task_name} (256x256)')
            plt.grid(alpha=0.3)
            plt.legend()
            
            save_path = output_dir / f'training_val_loss_{self.task_name}_256.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"📈 Loss 图已保存: {save_path}")
        except Exception as e:
            print(f"⚠️ 保存 Loss 图失败: {e}")

    def train(self, train_loader, val_loader):
        """训练循环 - 适配256x256"""
        print("🚀 开始256x256训练...")
        print(f"📏 输入分辨率: 256x256")
        print(f"🎯 任务: {self.task_name}")
        print(f"💡 优势: 相比512x512，训练速度更快，内存需求更低")
        
        train_losses = []
        val_losses = []
        
        # 初始验证
        print("\n🔍 进行初始验证...")
        initial_val_loss = self.validate(val_loader)
        print(f"初始验证损失: {initial_val_loss:.6f}")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch}/{self.config['num_epochs']} 完成")
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss: {val_loss:.6f}")
            print(f"   LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            print(f"{'='*60}")
            
            # 早停和最佳模型保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, self.config['task_name'], is_best=True)
                print(f"🎉 新的最佳验证损失: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"⏳ 早停计数: {self.patience_counter}/{self.max_patience}")
            
            # 定期保存
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, self.config['task_name'])
            
            # 早停检查
            if self.patience_counter >= self.max_patience:
                print(f"🛑 早停触发！在 epoch {epoch} 停止训练")
                break
            
            # 清理显存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                if epoch % 5 == 0:  # 每5个epoch记录一次内存
                    self._log_gpu_memory(f"End of Epoch {epoch}")
        
        # 保存最终模型和损失曲线
        self.save_checkpoint(self.config['num_epochs'], self.config['task_name'])
        self.plot_and_save_losses(train_losses, val_losses)
        
        print(f"\n🏁 训练完成！最佳验证损失: {self.best_val_loss:.6f}")

def main():
    # 256x256配置 - 立即改进的参数
    config = {
        'learning_rate': 5e-5,      # 立即改进：提高学习率
        'num_epochs': 40,
        'batch_size': 4,
        'save_interval': 5,
        'accumulation_steps': 2,
        'weight_decay': 1e-4,       # 立即改进：减小权重衰减
        'patience': 12,
        'warmup_steps': 500,        # 立即改进：添加warmup
    }

    tasks = [
        {'name': 'move_object', 'display': '移动物体'},
        {'name': 'drop_object', 'display': '掉落物体'},
        {'name': 'cover_object', 'display': '覆盖物体'}
    ]

    for task in tasks:
        task_name = task['name']
        print(f"\n{'='*60}")
        print(f"🚀 开始训练任务: {task['display']}")
        print(f"📏 分辨率: 256x256")
        print(f"{'='*60}")
        
        task_config = config.copy()
        task_config['task_name'] = task_name
        task_config['output_dir'] = f'training_results_{task_name}_256'
        
        # 加载数据 - 使用256数据集
        try:
            train_loader, val_loader, test_loader = create_task_specific_loaders(
                task_name=task_name,
                batch_size=task_config['batch_size'],
                data_path="processed_data_256"  # 修改为256数据集路径
            )
        except Exception as e:
            print(f"❌ 跳过任务 {task_name}: {e}")
            continue

        # 使用1000个样本
        max_samples = 1000
        if max_samples is not None:
            orig_train_ds = train_loader.dataset
            orig_val_ds = val_loader.dataset
            orig_test_ds = test_loader.dataset
            combined = ConcatDataset([orig_train_ds, orig_val_ds, orig_test_ds])
            total = len(combined)
            
            if max_samples > total:
                print(f"⚠️ 请求的样本数 {max_samples} 超过可用样本 {total}，使用全部样本")
                max_samples = total

            print(f"📊 从总样本 {total} 中随机抽取 {max_samples} 个（按 8:1:1 划分）")

            generator = torch.Generator()
            generator.manual_seed(42)
            perm = torch.randperm(total, generator=generator)[:max_samples].tolist()

            n_train = int(max_samples * 0.8)
            n_val = int(max_samples * 0.1)
            n_test = max_samples - n_train - n_val

            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train + n_val]
            test_idx = perm[n_train + n_val:]

            small_train_ds = Subset(combined, train_idx)
            small_val_ds = Subset(combined, val_idx)
            small_test_ds = Subset(combined, test_idx)

            train_loader = DataLoader(small_train_ds, batch_size=task_config['batch_size'], 
                                    shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(small_val_ds, batch_size=task_config['batch_size'], 
                                  shuffle=False, num_workers=4, pin_memory=True)

            print(f"✅ 数据集分配: train={len(small_train_ds)} val={len(small_val_ds)}")

        if len(train_loader) == 0:
            print(f"⚠️ 任务 {task_name} 无训练数据，跳过")
            continue

        # 初始化并训练
        try:
            trainer = ControlNet256Trainer(task_config)
            trainer.train(train_loader, val_loader)
        except Exception as e:
            print(f"❌ 训练任务 {task_name} 失败: {e}")
            continue

if __name__ == "__main__":
    import math  # 添加math导入用于学习率调度器
    main()