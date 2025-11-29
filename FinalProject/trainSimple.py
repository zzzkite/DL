#!/usr/bin/env python3
"""
ControlNet 1.1 简易版训练脚本 - Baseline
逻辑：输入第20帧的Canny边缘 -> 预测第25帧
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

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from loaderData import create_task_specific_loaders
    
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

# ❌ 删除了复杂的 EnhancedTemporalFeatureExtractor 类，直接用 OpenCV 处理

class SimpleControlNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_name = config.get('task_name', 'unknown_task')
        
        print(f"🚀 使用设备: {self.device} (RTX 3090 Mode)")
        print(f"🎯 任务: {self.task_name} - 简化版: 第20帧(Canny) → 预测第25帧")
        
        # 梯度缩放器
        self.scaler = torch.amp.GradScaler('cuda')
        # 显存监控与清理设置
        self.cleanup_steps = config.get('cleanup_steps', 20)  # 每多少个 batch 清理一次显存
        self.mem_log_steps = config.get('mem_log_steps', 10)  # 每多少个 batch 打印显存信息
        self.enable_mem_logging = config.get('enable_mem_logging', True)
        
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
        
        # 3. 冻结参数 (微调的核心)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(True) # 只训练 ControlNet
        
        # 打印参数量
        trainable_params = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
        print(f"✅ 模型初始化完成，可训练参数: {trainable_params:,}")

    def load_controlnet(self):
        """加载ControlNet模型"""
        try:
            if CONTROLNET_AVAILABLE:
                # 优先加载 Canny 预训练权重 (这非常适合只有250个数据的情况)
                controlnet_dir = Path("ControlNet-v1-1")
                model_path = controlnet_dir / "control_sd15_canny.pth"
                config_path = controlnet_dir / "cldm_v15.yaml"
                
                if model_path.exists():
                    print(f"📂 加载预训练权重: {model_path}")
                    model = create_model(str(config_path)).to(self.device)
                    model.load_state_dict(load_state_dict(str(model_path), location='cpu'))
                    return model.control_model
                else:
                    print("⚠️ 未找到预训练权重，将尝试从UNet初始化 (Scratch)")
            
            # 备选方案：从 UNet 初始化 (如果没有下载权重)
            from diffusers import ControlNetModel
            print("🆕 从 UNet 复制权重初始化 ControlNet")
            controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=3)
            return controlnet
                
        except Exception as e:
            print(f"❌ 加载ControlNet失败: {e}")
            raise

    # ----- GPU 内存监控/清理助手 -----
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

    def _cleanup_and_gc(self):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    
    def setup_optimizers(self):
        """优化器配置"""
        self.optimizer = optim.AdamW(
            self.controlnet.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=1e-2
        )
        # 简单的余弦退火
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['num_epochs']
        )

    def get_canny_edges(self, image_tensor):
        """
        将 Tensor 图像转换为 Canny 边缘图
        输入: (B, 3, H, W) 范围 [-1, 1] 或 [0, 1]
        输出: (B, 3, H, W) 范围 [0, 1]
        """
        # 1. 转换为 numpy (B, H, W, 3) 范围 [0, 255]
        # 假设输入已经是 [0, 1]
        images_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        edges_list = []
        for i in range(images_np.shape[0]):
            img_gray = cv2.cvtColor(images_np[i], cv2.COLOR_RGB2GRAY)
            # 自适应阈值 Canny
            v = np.median(img_gray)
            lower = int(max(0, 0.66 * v))
            upper = int(min(255, 1.33 * v))
            edge = cv2.Canny(img_gray, lower, upper)
            
            # 扩展回 3 通道并归一化到 [0, 1]
            edge = np.stack([edge]*3, axis=-1)
            edges_list.append(edge)
            
        edges_np = np.stack(edges_list)
        edges_tensor = torch.from_numpy(edges_np).float() / 255.0
        return edges_tensor.permute(0, 3, 1, 2).to(self.device)
    
    def train_epoch(self, train_loader, epoch):
        self.controlnet.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            if batch is None: continue
            
            # 1. 准备数据
            # input_frames: (B, 20, 3, H, W)
            # 我们只取第 20 帧 (Index -1) 作为输入条件
            current_frame_20 = batch['input_frames'][:, -1].to(self.device) 
            target_frame_25 = batch['target_frame'].to(self.device)
            text_descriptions = batch.get('label_text', ['interaction'] * len(current_frame_20))
            
            with torch.amp.autocast('cuda'):
                # 2. VAE 编码目标图 (Frame 25) -> Latents
                # 图像需要归一化到 [-1, 1]
                target_latents = self.vae.encode(target_frame_25 * 2.0 - 1.0).latent_dist.sample()
                target_latents = target_latents * self.vae.config.scaling_factor
                
                # 3. CLIP 编码文本
                inputs = self.tokenizer(text_descriptions, max_length=77, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]
                
                # 4. 准备 ControlNet 条件 (Frame 20 -> Canny)
                # 假设 data loader 出来的图像是 [0, 1]，如果不是请调整
                control_cond = self.get_canny_edges(current_frame_20)
                
                # 5. 加噪
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (target_latents.shape[0],), device=self.device).long()
                noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
                
                # 6. 前向传播
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_cond,
                    return_dict=False,
                )
                
                # 7. UNet 预测
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                loss = F.mse_loss(noise_pred, noise)

            # 8. 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印 loss
            if num_batches % 10 == 0:
                print(f"Epoch {epoch} | Batch {num_batches} | Loss: {loss.item():.4f}")

            # 定期打印显存并清理临时张量以避免爆显存
            if self.device.type == 'cuda' and self.enable_mem_logging and (num_batches % self.mem_log_steps == 0):
                self._log_gpu_memory(f"Epoch{epoch} Batch{num_batches} mid-step")

            # 删除大张量引用，释放 Python 层持有的引用
            try:
                del target_latents, encoder_hidden_states, control_cond, noise, timesteps, noisy_latents
                del down_block_res_samples, mid_block_res_sample, noise_pred
            except Exception:
                pass

            # 每隔 cleanup_steps 触发 empty_cache + gc
            if self.device.type == 'cuda' and ((num_batches) % self.cleanup_steps == 0):
                self._cleanup_and_gc()
                if self.enable_mem_logging:
                    self._log_gpu_memory(f"After cleanup Epoch{epoch} Batch{num_batches}")

        return total_loss / num_batches if num_batches > 0 else 0

    def save_checkpoint(self, epoch, task_name):
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / f"controlnet_{task_name}_epoch_{epoch}.pth"
        torch.save(self.controlnet.state_dict(), save_path)
        print(f"💾 模型已保存: {save_path}")

    def plot_and_save_losses(self, train_losses, val_losses=None):
        """绘制训练/验证损失并保存到输出目录（同一图）"""
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(8, 6))
            epochs = list(range(1, len(train_losses) + 1))
            plt.plot(epochs, train_losses, marker='o', color='tab:blue', label='Train Loss')
            if val_losses is not None and len(val_losses) == len(train_losses):
                plt.plot(epochs, val_losses, marker='o', color='tab:orange', label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training / Validation Loss - {self.task_name}')
            plt.grid(alpha=0.3)
            plt.legend()
            save_path = output_dir / f'training_val_loss_{self.task_name}.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"📈 Loss 图已保存: {save_path}")
        except Exception as e:
            print(f"⚠️ 保存 Loss 图失败: {e}")

    def validate(self, val_loader):
        """验证模型并返回平均验证损失"""
        self.controlnet.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                # 准备数据
                current_frame_20 = batch['input_frames'][:, -1].to(self.device)
                target_frame_25 = batch['target_frame'].to(self.device)
                text_descriptions = batch.get('label_text', ['interaction'] * len(current_frame_20))

                with torch.amp.autocast('cuda'):
                    target_latents = self.vae.encode(target_frame_25 * 2.0 - 1.0).latent_dist.sample()
                    target_latents = target_latents * self.vae.config.scaling_factor

                    inputs = self.tokenizer(text_descriptions, max_length=77, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                    encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]

                    control_cond = self.get_canny_edges(current_frame_20)

                    noise = torch.randn_like(target_latents)
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (target_latents.shape[0],), device=self.device).long()
                    noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=control_cond,
                        return_dict=False,
                    )

                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    loss = F.mse_loss(noise_pred, noise)

                total_loss += loss.item()
                num_batches += 1

                # 定期清理以避免持有过多显存
                if self.device.type == 'cuda' and (num_batches % self.mem_log_steps == 0):
                    self._log_gpu_memory(f"Validate Batch {num_batches}")
                    self._cleanup_and_gc()

        avg = total_loss / num_batches if num_batches > 0 else 0.0
        return avg

    def train(self, train_loader, val_loader):
        print("🚀 开始 Simple Version 训练...")
        
        train_losses = []
        val_losses = []
        for epoch in range(1, self.config['num_epochs'] + 1):
            loss = self.train_epoch(train_loader, epoch)
            train_losses.append(loss)

            # 验证
            try:
                val_loss = self.validate(val_loader)
            except Exception as e:
                print(f"⚠️ 验证阶段出错: {e}")
                val_loss = 0.0
            val_losses.append(val_loss)

            self.lr_scheduler.step()
            print(f"=== Epoch {epoch} 完成 | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} ===")

            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, self.config['task_name'])

            # 每个 epoch 结束时再做一次全面的清理和显存记录
            if self.device.type == 'cuda':
                self._cleanup_and_gc()
                if self.enable_mem_logging:
                    self._log_gpu_memory(f"End of Epoch {epoch}")

        # 任务完成后保存训练/验证 Loss 曲线到输出目录
        try:
            self.plot_and_save_losses(train_losses, val_losses)
        except Exception as e:
            print(f"⚠️ 绘制/保存 Loss 曲线失败: {e}")

def main():
    # 基础配置
    config = {
        'learning_rate': 2e-5, # ControlNet 微调常用 LR
        'num_epochs': 50,      # 数据少，不需要太多epoch，或者根据loss情况早停
        'batch_size': 4,       # 3090 可以适当大一点
        'save_interval': 10,
    }

    # 任务列表 (保持原代码结构)
    tasks = [
        {'name': 'move_object', 'display': '移动物体'},
        {'name': 'drop_object', 'display': '掉落物体'},
        {'name': 'cover_object', 'display': '覆盖物体'}
    ]

    for task in tasks:
        task_name = task['name']
        print(f"\n🚀 开始训练任务: {task['display']}")
        
        task_config = config.copy()
        task_config['task_name'] = task_name
        task_config['output_dir'] = f'training_results_{task_name}_simple'
        
        # 加载数据
        try:
            # 复用你们原来的 loader 接口
            train_loader, val_loader, test_loader = create_task_specific_loaders(
                task_name=task_name,
                batch_size=task_config['batch_size'],
                data_path="processed_data"
            )
        except Exception as e:
            print(f"跳过任务 {task_name}: {e}")
            continue

        # 如果用户希望只用部分样本进行快速验证，可以通过 config['max_samples'] 控制
        # 它会从 train/val/test 三个原始分区的联合集合中随机抽取总计 max_samples 个样本，
        # 并按照 8:1:1 的比例重新划分为新训练/验证/测试集。
        max_samples = task_config.get('max_samples', 1000)  # e.g., 1000
        if max_samples is not None:
            # 合并原始三个 dataset
            orig_train_ds = train_loader.dataset
            orig_val_ds = val_loader.dataset
            orig_test_ds = test_loader.dataset
            combined = ConcatDataset([orig_train_ds, orig_val_ds, orig_test_ds])
            total = len(combined)
            if max_samples > total:
                print(f"请求的样本数 {max_samples} 超过可用样本 {total}，将使用全部样本")
                max_samples = total

            print(f"从总样本 {total} 中随机抽取 {max_samples} 个用于快速验证（按 8:1:1 划分）")

            # 随机选择索引
            generator = torch.Generator()
            generator.manual_seed(task_config.get('random_seed', 42))
            perm = torch.randperm(total, generator=generator)[:max_samples].tolist()

            # 按 8:1:1 划分
            n_train = int(max_samples * 0.8)
            n_val = int(max_samples * 0.1)
            n_test = max_samples - n_train - n_val

            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train + n_val]
            test_idx = perm[n_train + n_val:]

            # 创建 Subset dataset 并 DataLoader
            small_train_ds = Subset(combined, train_idx)
            small_val_ds = Subset(combined, val_idx)
            small_test_ds = Subset(combined, test_idx)

            train_loader = DataLoader(small_train_ds, batch_size=task_config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(small_val_ds, batch_size=task_config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
            test_loader = DataLoader(small_test_ds, batch_size=task_config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

            print(f"=> 快速验证集样本分配: train={len(small_train_ds)} val={len(small_val_ds)} test={len(small_test_ds)}")

        if len(train_loader) == 0:
            continue

        # 初始化并训练
        trainer = SimpleControlNetTrainer(task_config)
        trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()