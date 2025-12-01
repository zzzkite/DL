import torch
import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ================= 1. 配置区域 =================
class Config:
    TEST_CSV = "./processed_data_512/metadata/test_samples.csv"
    DATA_ROOT = "./processed_data_512"
    
    # 修改为你的训练好的ControlNet权重路径
    CONTROLNET_PATH = "./training_results_move_object_512/controlnet_move_object_best.pth"
    
    # 修改：明确指定本地模型路径
    SD_LOCAL_PATH = "./stable-diffusion-v1-5"
    
    OUTPUT_DIR = "./evaluation_results"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 2. 工具函数 =================
def npy_to_pil(npy_path, is_input_stack=False):
    """读取 .npy 文件并转换为 PIL Image"""
    try:
        data = np.load(npy_path)

        if is_input_stack and data.ndim == 4:
            data = data[-1]

        if data.shape[0] == 3:
            data = np.transpose(data, (1, 2, 0))

        if data.min() < 0:
            data = (data + 1) / 2 * 255
        elif data.max() <= 1.0:
            data = data * 255

        data = np.clip(data, 0, 255).astype(np.uint8)
        return Image.fromarray(data)

    except Exception as e:
        print(f"❌ 读取 NPY 失败 {npy_path}: {e}")
        return None

def calculate_metrics(img1, img2):
    """计算 PSNR 和 SSIM"""
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        score_psnr = psnr(img1, img2, data_range=255)
        score_ssim = ssim(img1, img2, channel_axis=2, data_range=255, win_size=3)
        return score_psnr, score_ssim
    except ImportError:
        print("⚠️  skimage不可用，使用OpenCV计算PSNR")
        score_psnr = cv2.PSNR(img1, img2)
        return score_psnr, 0.0

def load_model():
    """加载模型 - 完全使用本地文件，正确的方式"""
    print("📦 正在加载模型...")
    
    # 方法1：尝试使用cldm库加载（如果可用）
    try:
        from cldm.model import create_model, load_state_dict
        print("✅ 使用cldm加载ControlNet")
        
        # 使用canny配置创建ControlNet
        controlnet_model = create_model("./ControlNet-v1-1/control_v11p_sd15_canny.yaml").cpu()
        controlnet_model.load_state_dict(load_state_dict("./ControlNet-v1-1/control_v11p_sd15_canny.pth", location='cpu'))
        controlnet = controlnet_model.control_model
        
        # 加载你训练好的权重
        print(f"🔄 加载训练权重: {Config.CONTROLNET_PATH}")
        trained_checkpoint = torch.load(Config.CONTROLNET_PATH, map_location='cpu')
        if 'model_state_dict' in trained_checkpoint:
            controlnet.load_state_dict(trained_checkpoint['model_state_dict'])
        else:
            controlnet.load_state_dict(trained_checkpoint)
            
    except ImportError:
        # 方法2：如果cldm不可用，使用更简单的方法
        print("⚠️  cldm不可用，使用备用方案")
        
        # 直接从预训练初始化ControlNet，然后加载权重
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
        
        # 加载你训练好的权重
        print(f"🔄 加载训练权重: {Config.CONTROLNET_PATH}")
        trained_checkpoint = torch.load(Config.CONTROLNET_PATH, map_location='cpu')
        
        # 处理权重文件
        if 'model_state_dict' in trained_checkpoint:
            state_dict = trained_checkpoint['model_state_dict']
        elif 'controlnet_state_dict' in trained_checkpoint:
            state_dict = trained_checkpoint['controlnet_state_dict']
        else:
            state_dict = trained_checkpoint
        
        # 移除可能的前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('controlnet.'):
                new_k = k.replace('controlnet.', '')
            elif k.startswith('module.'):
                new_k = k.replace('module.', '')
            else:
                new_k = k
            new_state_dict[new_k] = v
        
        # 加载权重
        missing_keys, unexpected_keys = controlnet.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  意外的键: {unexpected_keys}")

    # 加载Stable Diffusion管道 - 使用本地文件
    print("🔄 加载Stable Diffusion管道...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        Config.SD_LOCAL_PATH,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(Config.DEVICE)
    
    print("✅ 模型加载完成")
    return pipe

# ================= 3. 主程序 =================
def main():
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"📖 读取测试列表: {Config.TEST_CSV}")
    try:
        df = pd.read_csv(Config.TEST_CSV)
    except Exception as e:
        print(f"❌ 无法读取 CSV: {e}")
        return

    pipe = load_model()
    metrics = {'ours_psnr': [], 'ours_ssim': [], 'baseline_psnr': [], 'baseline_ssim': []}

    print(f"🚀 开始评估 {len(df)} 个样本...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            video_id = str(row['video_id'])
            category = row['category']

            folder_path = os.path.join(Config.DATA_ROOT, "frames", category)
            input_npy_path = os.path.join(folder_path, f"{video_id}_input.npy")
            target_npy_path = os.path.join(folder_path, f"{video_id}_target.npy")

            prompt = row['label']

            if not os.path.exists(input_npy_path):
                continue

            # 读取NPY并转为图片
            cond_img_pil = npy_to_pil(input_npy_path, is_input_stack=True)
            if cond_img_pil is None: continue
            cond_img_pil = cond_img_pil.resize((512, 512))
            cond_np = np.array(cond_img_pil)

            gt_img_pil = npy_to_pil(target_npy_path, is_input_stack=False)
            if gt_img_pil is None: continue
            gt_img_pil = gt_img_pil.resize((512, 512))
            gt_np = np.array(gt_img_pil)

            # 制作Canny控制图
            img_cv = cv2.cvtColor(cond_np, cv2.COLOR_RGB2BGR)
            canny = cv2.Canny(img_cv, 100, 200)
            canny = np.stack([canny] * 3, axis=2)
            canny_pil = Image.fromarray(canny)

            # 模型推理
            with torch.no_grad():
                generated_image = pipe(
                    prompt,
                    image=canny_pil,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=1.0
                ).images[0]

            gen_np = np.array(generated_image)

            # 计算指标
            p_ours, s_ours = calculate_metrics(gt_np, gen_np)
            metrics['ours_psnr'].append(p_ours)
            metrics['ours_ssim'].append(s_ours)

            p_base, s_base = calculate_metrics(gt_np, cond_np)
            metrics['baseline_psnr'].append(p_base)
            metrics['baseline_ssim'].append(s_base)

            # 保存可视化（前20张）
            if idx < 20:
                w, h = 512, 512
                vis = Image.new('RGB', (w * 3, h))
                vis.paste(cond_img_pil, (0, 0))  # Input
                vis.paste(generated_image, (w, 0))  # Ours
                vis.paste(gt_img_pil, (w * 2, 0))  # GT

                from PIL import ImageDraw
                draw = ImageDraw.Draw(vis)
                draw.text((10, 10), f"Ours PSNR: {p_ours:.2f}", fill=(255, 0, 0))

                vis.save(f"{Config.OUTPUT_DIR}/sample_{idx}_{video_id}.jpg")

        except Exception as e:
            print(f"⚠️ 样本 {video_id} 错误: {e}")
            continue

    # 打印报告
    print("\n" + "=" * 40)
    print("📊 评估结果")
    print("=" * 40)

    if len(metrics['ours_psnr']) == 0:
        print("❌ 评估失败：没有找到任何有效样本")
        return

    avg_ours_psnr = np.mean(metrics['ours_psnr'])
    avg_ours_ssim = np.mean(metrics['ours_ssim'])
    avg_base_psnr = np.mean(metrics['baseline_psnr'])
    avg_base_ssim = np.mean(metrics['baseline_ssim'])

    print(f"{'Method':<20} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 46)
    print(f"{'Baseline (Copy)':<20} | {avg_base_psnr:<10.4f} | {avg_base_ssim:<10.4f}")
    print(f"{'Ours (Model)':<20} | {avg_ours_psnr:<10.4f} | {avg_ours_ssim:<10.4f}")
    print("-" * 46)

if __name__ == "__main__":
    main()