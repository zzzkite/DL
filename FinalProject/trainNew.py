#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from accelerate import Accelerator
import cv2
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, \
    StableDiffusionControlNetPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 防止在服务器上画图报错
plt.switch_backend('agg')


# 如果仓库中没有独立的 dataset.py，我们在这里提供一个小型实现以兼容训练脚本
class SimpleVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, tokenizer, size=512, is_training=True, split='train'):
        """从 metadata CSV 加载样本并返回训练需要的字段。
        期望 metadata CSV 位于 data_root/metadata/{train,val,test}_samples.csv，
        每行包含 input_frames_path 和 target_frame_path 以及 label 字段。
        """
        import csv
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.is_training = is_training

        split_file = 'train_samples.csv' if is_training else 'val_samples.csv'
        meta_path = os.path.join(data_root, 'metadata', split_file)
        if not os.path.exists(meta_path):
            # allow test split usage
            meta_path = os.path.join(data_root, 'metadata', 'test_samples.csv')

        self.rows = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    self.rows.append(r)
        else:
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

    def __len__(self):
        return len(self.rows)

    def _load_npy(self, path):
        # path may be absolute or relative; use cwd as base
        full = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        arr = np.load(full)
        return arr

    def _make_canny(self, frame):
        # frame: H,W,3 uint8 or float in [0,1]
        if frame.dtype != np.uint8:
            frame_u8 = (frame * 255).astype(np.uint8)
        else:
            frame_u8 = frame
        gray = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY)
        # auto sigma-based thresholds
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edge = cv2.Canny(gray, lower, upper)
        edge = np.stack([edge] * 3, axis=-1).astype(np.uint8)
        edge = edge.astype(np.float32) / 255.0
        # to CHW
        edge = np.transpose(edge, (2, 0, 1))
        return edge

    def __getitem__(self, idx):
        row = self.rows[idx]
        inp_path = row['input_frames_path']
        tgt_path = row['target_frame_path']

        inp = self._load_npy(inp_path)  # (T,H,W,C)
        tgt = self._load_npy(tgt_path)  # (H,W,C)

        # last frame as condition
        last = inp[-1]  # (H,W,C)

        # normalize
        if last.dtype != np.float32 and last.max() > 1.0:
            cond = last.astype(np.float32) / 255.0
        else:
            cond = last.astype(np.float32)

        if tgt.dtype != np.float32 and tgt.max() > 1.0:
            tgt_f = tgt.astype(np.float32) / 255.0
        else:
            tgt_f = tgt.astype(np.float32)

        # conditioning: C,H,W in [0,1]
        cond_chw = self._make_canny(cond)

        # pixel_values (target) should be in [-1,1]
        pv = np.transpose(tgt_f, (2, 0, 1))  # C,H,W
        pixel_values = pv * 2.0 - 1.0

        # also keep original RGB frames for visualization (H,W,C) in [0,255] uint8
        def _to_uint8_image(arr):
            # arr: numpy array HWC, may be float in [0,1] or [0,255], or uint8
            if isinstance(arr, np.ndarray):
                if arr.dtype == np.uint8:
                    return arr
                # avoid division/multiplication mistakes: detect range
                amin = float(np.min(arr))
                amax = float(np.max(arr))
                if amax <= 1.0 + 1e-6:
                    img = (arr * 255.0).round().astype(np.uint8)
                    return img
                # if values already in 0..255 range but float, clip then convert
                img = np.clip(arr, 0, 255).round().astype(np.uint8)
                return img
            else:
                return np.array(arr).astype(np.uint8)

        last_rgb = _to_uint8_image(last)
        # prefer original target array 'tgt' for visualization (not the normalized tgt_f)
        tgt_rgb = _to_uint8_image(tgt)

        prompt = row.get('label', row.get('template', 'interaction'))
        # tokenize
        toks = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids = toks.input_ids.squeeze(0)

        # convert to torch tensors
        import torch as _torch
        return {
            'pixel_values': _torch.from_numpy(pixel_values).float(),
            'conditioning_pixel_values': _torch.from_numpy(cond_chw).float(),
            'input_ids': input_ids.long(),
            'prompt': prompt,
            'last_frame_rgb': last_rgb,  # H,W,C uint8
            'target_frame_rgb': tgt_rgb  # H,W,C uint8
        }

# ================= 最终版配置 (带智能挑选样本) =================
# 路径配置（优先使用项目内本地模型）
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "stable-diffusion-v1-5")
# ControlNet 本地目录（如果你已经转换为 diffusers 格式则可直接使用），否则会回退到 hub
CONTROLNET_DIR = os.path.join(BASE_DIR, "ControlNet-v1-1")
DATA_DIR = "./augmented_data_512"
OUTPUT_DIR = "./final_model_output_viz_v5"  # 改个名字，防止覆盖之前的

# 训练参数
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 3e-5
NUM_EPOCHS = 50
SAVE_INTERVAL = 10


# =======================================================

def save_loss_plot(loss_history, output_dir):
    """绘制并保存训练/验证 Loss 曲线。
    参数可以是每个 epoch 的平均 loss 列表。"""
    train_losses, val_losses = loss_history if isinstance(loss_history, tuple) else (loss_history, None)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    if val_losses is not None:
        plt.plot(val_losses, label="Val Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step,
                   validation_samples):
    """生成预览图用于验证"""
    print(f"\n🖼️ 正在生成第 {step} 步的验证预览图 (Selected Best Samples)...")

    # 创建临时的 Pipeline
    # 优先使用项目内的 MODEL_DIR（如果存在），否则使用 hub
    pipeline_model_dir = MODEL_DIR if os.path.exists(MODEL_DIR) else "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pipeline_model_dir,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=accelerator.unwrap_model(controlnet),
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=os.path.exists(MODEL_DIR)
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device)

    for idx, sample in enumerate(validation_samples):
        # 1. 原始第20帧（last frame）
        last_rgb = sample.get('last_frame_rgb')
        if last_rgb is None:
            cond_tensor = sample["conditioning_pixel_values"]
            cond_img = cond_tensor.permute(1, 2, 0).cpu().numpy()
            last_rgb = (cond_img * 255).astype(np.uint8)
        last_pil = Image.fromarray(last_rgb)

        # 2. 边缘检测图（conditioning）
        cond_tensor = sample["conditioning_pixel_values"]
        cond_img = cond_tensor.permute(1, 2, 0).cpu().numpy()
        cond_img = (cond_img * 255).astype(np.uint8)
        cond_pil = Image.fromarray(cond_img)

        # 3. 生成预测第25帧
        prompt = sample["prompt"]
        generator = torch.Generator(device=accelerator.device).manual_seed(42)
        with torch.autocast(device_type=accelerator.device.type):
            pred = pipeline(
                prompt,
                image=cond_pil,
                num_inference_steps=20,
                generator=generator
            ).images[0]
        # 标准化预测图：确保为 RGB uint8 numpy 并与其他图一致
        if not isinstance(pred, Image.Image):
            pred_pil = Image.fromarray(np.array(pred).astype(np.uint8)).convert('RGB')
        else:
            pred_pil = pred.convert('RGB')

        # 4. 真实第25帧
        tgt_rgb = sample.get('target_frame_rgb')
        if tgt_rgb is None:
            gt_tensor = sample["pixel_values"]
            gt_img = ((gt_tensor + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
            gt_img = np.transpose(gt_img, (1, 2, 0))
            tgt_pil = Image.fromarray(gt_img)
        else:
            tgt_pil = Image.fromarray(tgt_rgb)

        # 统一尺寸：以 last_pil 的尺寸为基准，调整其他图像
        w, h = last_pil.size
        def _ensure_size(img, size):
            if img.size != size:
                return img.resize(size, resample=Image.BILINEAR)
            return img

        last_img = last_pil.convert('RGB')
        cond_img_small = cond_pil.convert('RGB')
        pred_img = _ensure_size(pred_pil, (w, h))
        tgt_img = _ensure_size(tgt_pil.convert('RGB'), (w, h))

        combined = Image.new("RGB", (w * 4, h))
        combined.paste(last_img, (0, 0))
        combined.paste(cond_img_small, (w, 0))
        combined.paste(pred_img, (w * 2, 0))
        combined.paste(tgt_img, (w * 3, 0))

        save_path = os.path.join(OUTPUT_DIR, f"val_epoch_{step}_sample_{idx}.jpg")
        combined.save(save_path)

    del pipeline
    torch.cuda.empty_cache()
    print("✅ 优质预览图已保存")


def compute_validation_loss(vae, text_encoder, unet, controlnet, noise_scheduler, validation_samples, accelerator):
    """根据当前模型计算 validation samples 的平均 MSE loss。返回 float。"""
    if len(validation_samples) == 0:
        return 0.0

    # 合并成 batch
    # Use the device/dtype prepared by accelerator:
    pixel_vals = torch.stack([s['pixel_values'] for s in validation_samples], dim=0)
    conds = torch.stack([s['conditioning_pixel_values'] for s in validation_samples], dim=0)
    # move to accelerator device
    pixel_vals = pixel_vals.to(accelerator.device)
    conds = conds.to(accelerator.device)
    # tokenization -> input_ids may be tensor or list
    if isinstance(validation_samples[0].get('input_ids', None), torch.Tensor):
        input_ids = torch.stack([s['input_ids'] for s in validation_samples], dim=0).to(accelerator.device)
    else:
        input_ids = None

    vae = vae
    unet = unet
    controlnet = controlnet
    text_encoder = text_encoder

    with torch.no_grad():
        # Use accelerator.autocast to respect mixed-precision without manual casting
        with accelerator.autocast():
            # encode latents
            latents = vae.encode(pixel_vals).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0] if input_ids is not None else None

            down, mid = controlnet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=conds,
                return_dict=False
            )

            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down, mid_block_additional_residual=mid
            ).sample

            # ensure noise dtype matches prediction dtype
            noise = noise.to(dtype=noise_pred.dtype)

            loss = F.mse_loss(noise_pred, noise, reduction='mean')
            return float(loss.item())


def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        mixed_precision="fp16"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    print("📦 加载预训练模型 (优先本地)...")
    # Tokenizer / text encoder / vae / unet 优先从本地 MODEL_DIR 加载
    use_local = os.path.exists(MODEL_DIR)
    try:
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR if use_local else "runwayml/stable-diffusion-v1-5",
                                                   subfolder="tokenizer", local_files_only=use_local)
        text_encoder = CLIPTextModel.from_pretrained(MODEL_DIR if use_local else "runwayml/stable-diffusion-v1-5",
                                                     subfolder="text_encoder", local_files_only=use_local)
        vae = AutoencoderKL.from_pretrained(MODEL_DIR if use_local else "runwayml/stable-diffusion-v1-5",
                                            subfolder="vae", local_files_only=use_local)
        unet = UNet2DConditionModel.from_pretrained(MODEL_DIR if use_local else "runwayml/stable-diffusion-v1-5",
                                                   subfolder="unet", local_files_only=use_local)
    except Exception as e:
        print(f"❌ 加载 Stable Diffusion 组件失败: {e}")
        return

    # ControlNet: 优先尝试本地 diffusers 格式目录，再尝试 hub 加载
    controlnet = None
    try:
        if os.path.exists(CONTROLNET_DIR):
            try:
                print(f"🔁 尝试从本地 ControlNet 目录加载: {CONTROLNET_DIR}")
                controlnet = ControlNetModel.from_pretrained(CONTROLNET_DIR, local_files_only=True)
            except Exception:
                controlnet = None

        if controlnet is None:
            print("🌐 回退到在线 hub 加载 ControlNet: lllyasviel/sd-controlnet-canny")
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", local_files_only=False)
    except Exception as e:
        print(f"⚠️ 无法加载 ControlNet（本地或在线）: {e}")
        return

    # 冻结
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # controlnet.enable_gradient_checkpointing()
    unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3,
        betas=(0.9, 0.999),  # 调整beta参数
        eps=1e-8
    )

    # 2. 数据加载
    print(f"📂 读取数据集: {DATA_DIR}")
    dataset = SimpleVideoDataset(DATA_DIR, tokenizer, size=512, is_training=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # === 🔍 智能挑选最好的验证样本 (核心修改) ===
    validation_samples = []
    print("🔍 正在寻找最有代表性的验证样本 (Moving & Dropping)...")
    # 选择更多样本用于预览（默认 4 个）
    VALIDATION_SAMPLE_COUNT = 10
    selected_indices = []
    found_move = False
    found_drop = False

    start_search = np.random.randint(0, max(1, len(dataset) // 3))
    # 先尝试挑选具有代表性的样本
    for i in range(start_search, len(dataset)):
        if len(selected_indices) >= VALIDATION_SAMPLE_COUNT:
            break
        item = dataset[i]
        ids = item["input_ids"]
        decoded_prompt = tokenizer.decode(ids, skip_special_tokens=True).lower()

        if not found_drop and "dropping" in decoded_prompt:
            selected_indices.append(i)
            found_drop = True
            print(f"   ✅ 选中高质量样本 B (Drop): ID {i} | Prompt: {decoded_prompt}")
            continue

        if not found_move and "moving" in decoded_prompt and ("left" in decoded_prompt or "right" in decoded_prompt):
            selected_indices.append(i)
            found_move = True
            print(f"   ✅ 选中高质量样本 A (Move): ID {i} | Prompt: {decoded_prompt}")
            continue

    # 补齐其余样本
    rng = np.random.default_rng()
    while len(selected_indices) < VALIDATION_SAMPLE_COUNT:
        rnd_idx = int(rng.integers(0, len(dataset)))
        if rnd_idx not in selected_indices:
            selected_indices.append(rnd_idx)

    # 加载样本并保留 input_ids
    for i in selected_indices:
        item = dataset[i]
        decoded_prompt = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        validation_samples.append({
            "pixel_values": item["pixel_values"],
            "conditioning_pixel_values": item["conditioning_pixel_values"],
            "input_ids": item["input_ids"],
            "prompt": decoded_prompt,
            "last_frame_rgb": item.get('last_frame_rgb'),
            "target_frame_rgb": item.get('target_frame_rgb')
        })
    # ==========================================

    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else "runwayml/stable-diffusion-v1-5", subfolder="scheduler", local_files_only=os.path.exists(MODEL_DIR))

    # 使用 Accelerator 一次性准备所有模型/优化器/数据加载器，避免手动 .to(...) 导致的 dtype/device 不一致
    controlnet, vae, unet, text_encoder, optimizer, dataloader = accelerator.prepare(
        controlnet, vae, unet, text_encoder, optimizer, dataloader
    )

    # 训练循环
    global_step = 0
    loss_history = []
    epoch_train_losses = []
    epoch_val_losses = []

    print(f"🚀 开始训练 (Smart Validation Enabled)! Epochs: {NUM_EPOCHS}")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}")

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(controlnet):
                # batch tensors are already prepared by Accelerator and on the correct device/dtype
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                cond = batch["conditioning_pixel_values"]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                          device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                down, mid = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond,
                    return_dict=False
                )
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down, mid_block_additional_residual=mid
                ).sample

                loss = F.mse_loss(noise_pred, noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            loss_val = loss.item()
            total_loss += loss_val
            loss_history.append(loss_val)
            progress_bar.set_postfix(loss=loss_val)

        avg_loss = total_loss / len(dataloader)
        print(f"🏁 Epoch {epoch + 1} Avg Loss: {avg_loss:.5f}")
        # 计算验证 loss（在每个 epoch 之后）
        try:
            val_loss = compute_validation_loss(vae, text_encoder, unet, controlnet, noise_scheduler, validation_samples, accelerator)
        except Exception as e:
            print(f"⚠️ 计算验证损失时出错: {e}")
            val_loss = None

        epoch_train_losses.append(avg_loss)
        if val_loss is not None:
            epoch_val_losses.append(val_loss)

        # 打印训练与验证 loss
        if val_loss is not None:
            print(f"    ▶️ Validation Loss: {val_loss:.5f}")

        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            # 保存模型
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch + 1}")
            accelerator.save_state(save_path)
            unwrapped = accelerator.unwrap_model(controlnet)
            unwrapped.save_pretrained(os.path.join(save_path, "controlnet"))
            print(f"💾 模型已保存: {save_path}")

            # 生成预览图 (这次是精选的样本)
            log_validation(
                vae, text_encoder, tokenizer, unet, controlnet,
                None, accelerator, torch.float16, epoch + 1, validation_samples
            )

            # 更新 Loss 曲线
            save_loss_plot((epoch_train_losses, epoch_val_losses), OUTPUT_DIR)


if __name__ == "__main__":
    main()