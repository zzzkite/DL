#!/usr/bin/env python3
"""
ControlNet 1.1 512x512 极稳训练脚本
优化点：超低学习率 + 大梯度累积 + 在线数据增强
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.optim as optim
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import time
from PIL import Image
from torchvision import transforms  # 引入 torchvision 做增强

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from loaderData512 import create_task_specific_loaders

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


class ControlNet512Trainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_name = config.get('task_name', 'unknown_task')

        print(f"🚀 使用设备: {self.device}")
        print(f"🎯 任务: {self.task_name}")
        print(f"📉 学习率: {config['learning_rate']} (低LR模式)")
        print(f"📦 等效BatchSize: {config['batch_size'] * config['accumulation_steps']}")

        self.best_val_loss = float('inf')
        self.scaler = torch.amp.GradScaler('cuda')

        # 定义数据增强 (仅用于训练集)
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

        self.setup_models()
        self.setup_optimizers()

    def setup_models(self):
        print("📦 初始化模型...")
        self.tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/tokenizer", local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained("stable-diffusion-v1-5/text_encoder",
                                                          local_files_only=True).to(self.device)
        self.vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/vae", local_files_only=True).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/unet", local_files_only=True).to(
            self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained("stable-diffusion-v1-5/scheduler", local_files_only=True)
        self.controlnet = self.load_controlnet().to(self.device)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(True)

    def load_controlnet(self):
        try:
            if CONTROLNET_AVAILABLE:
                controlnet_dir = Path("ControlNet-v1-1")
                model_path = controlnet_dir / "control_sd15_canny.pth"
                config_path = controlnet_dir / "cldm_v15.yaml"
                if model_path.exists():
                    model = create_model(str(config_path)).to(self.device)
                    model.load_state_dict(load_state_dict(str(model_path), location='cpu'))
                    return model.control_model
            from diffusers import ControlNetModel
            return ControlNetModel.from_unet(self.unet, conditioning_channels=3)
        except Exception as e:
            raise e

    def setup_optimizers(self):
        # 增加 weight_decay 到 5e-2 以增强正则化
        self.optimizer = optim.AdamW(
            self.controlnet.parameters(),
            lr=self.config.get('learning_rate', 5e-6),
            weight_decay=self.config.get('weight_decay', 5e-2),
            eps=1e-8
        )
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('num_epochs', 150),
            eta_min=1e-7
        )

    def prepare_images_for_vae(self, images):
        if torch.max(images) <= 1.0 and torch.min(images) >= 0.0:
            images = images * 2.0 - 1.0
        vae_sample_size = self.vae.config.sample_size
        if images.shape[-1] != vae_sample_size:
            images = F.interpolate(images, size=(vae_sample_size, vae_sample_size), mode='bilinear',
                                   align_corners=False)
        return images

    def get_canny_edges(self, image_tensor):
        # Canny 处理逻辑保持不变
        batch_size = image_tensor.shape[0]
        if torch.max(image_tensor) > 1.0:
            image_tensor = (image_tensor + 1.0) / 2.0

        images_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)

        edges_list = []
        for i in range(batch_size):
            img_gray = cv2.cvtColor(images_np[i], cv2.COLOR_RGB2GRAY)
            v = np.median(img_gray)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edge = cv2.Canny(img_gray, lower, upper)
            edge = np.stack([edge] * 3, axis=-1)
            edges_list.append(edge)

        edges_np = np.stack(edges_list)
        edges_tensor = torch.from_numpy(edges_np).float() / 255.0
        return edges_tensor.permute(0, 3, 1, 2).to(self.device)

    def apply_augmentation(self, images):
        """对 Tensor 图像应用数据增强 (B, 3, H, W) [0, 1]"""
        # 为了使用 torch transforms，我们需要确保形状正确
        augmented = self.aug_transform(images)
        return augmented

    def compute_loss(self, batch, training=True):
        current_frame_20 = batch['input_frames'][:, -1].to(self.device)
        target_frame_25 = batch['target_frame'].to(self.device)
        text_descriptions = batch.get('label_text', ['interaction'] * len(current_frame_20))

        # === 核心修改：仅在训练时对目标图应用数据增强 ===
        # 这就是你要的"归一化/正则化"效果，防止死记硬背
        if training:
            # 输入原本是 [0, 1] 或 [-1, 1]，这里假设 loader 出来是 [0, 1]
            # 如果是 [-1, 1] 先转回去
            if torch.min(target_frame_25) < 0:
                target_frame_25 = (target_frame_25 + 1) / 2.0

            target_frame_25 = self.apply_augmentation(target_frame_25)

            # 这里的 Input Condition (frame 20) 也可以稍微增强一点点，但最好保持结构
            # 暂时只增强 Target，增加生成的鲁棒性
        # ============================================

        target_frame_prepared = self.prepare_images_for_vae(target_frame_25)
        target_latents = self.vae.encode(target_frame_prepared).latent_dist.sample()
        target_latents = target_latents * self.vae.config.scaling_factor

        inputs = self.tokenizer(text_descriptions, max_length=77, padding="max_length", truncation=True,
                                return_tensors="pt").to(self.device)
        encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]

        control_cond = self.get_canny_edges(current_frame_20)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (target_latents.shape[0],),
                                  device=self.device).long()
        noise = torch.randn_like(target_latents)
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
        return loss

    def train_epoch(self, train_loader, epoch):
        self.controlnet.train()
        total_loss = 0
        num_batches = 0
        # ⬇️ 核心修改：提高累积步数，模拟大Batch Size
        accumulation_steps = self.config.get('accumulation_steps', 16)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue

            try:
                with torch.amp.autocast('cuda'):
                    loss = self.compute_loss(batch, training=True)

                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                loss_value = loss.item() * accumulation_steps
                total_loss += loss_value
                num_batches += 1

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss_value:.4f}")

            except Exception as e:
                print(f"Err: {e}")
                continue

        return total_loss / num_batches if num_batches > 0 else 0

    def validate(self, val_loader):
        self.controlnet.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                try:
                    with torch.amp.autocast('cuda'):
                        loss = self.compute_loss(batch, training=False)
                    total_loss += loss.item()
                    num_batches += 1
                except:
                    continue
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, epoch, task_name, is_best=False):
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"controlnet_{task_name}_best.pth" if is_best else f"controlnet_{task_name}_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.controlnet.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, output_dir / filename)
        print(f"💾 Saved: {filename}")

    def train(self, train_loader, val_loader):
        print(f"🚀 开始极稳模式训练 (LR={self.config['learning_rate']})")

        for epoch in range(1, self.config['num_epochs'] + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            self.lr_scheduler.step()

            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, self.config['task_name'], is_best=True)

            # 每20轮保存一次常规checkpoint，防止磁盘写满
            if epoch % 20 == 0:
                self.save_checkpoint(epoch, self.config['task_name'])


def main():
    # ================= 极稳参数配置 =================
    config = {
        'learning_rate': 5e-6,  # ⬇️ 极低学习率，防止震荡
        'num_epochs': 150,
        'batch_size': 2,  # 物理 Batch
        'accumulation_steps': 16,  # ⬆️ 梯度累积16次 -> 等效 Batch 32
        'weight_decay': 5e-2,  # ⬆️ 高正则化，防止过拟合
        'save_interval': 10,
    }
    # ==============================================

    tasks = [
        {'name': 'move_object', 'display': '移动物体'},
        {'name': 'drop_object', 'display': '掉落物体'},
        {'name': 'cover_object', 'display': '覆盖物体'}
    ]

    for task in tasks:
        task_name = task['name']
        print(f"\n🚀 任务: {task['display']}")
        task_config = config.copy()
        task_config['task_name'] = task_name
        task_config['output_dir'] = f'training_results_{task_name}_Low_512'

        try:
            train_loader, val_loader, test_loader = create_task_specific_loaders(
                task_name=task_name,
                batch_size=task_config['batch_size'],
                data_path="processed_data_512"
            )
        except Exception as e:
            continue

        # 采样部分代码省略，保持原样即可...
        # 建议直接用 train_loader 跑，不要太复杂的 subset 逻辑，除非数据真的太多

        if len(train_loader) == 0: continue

        trainer = ControlNet512Trainer(task_config)
        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()