#!/usr/bin/env python3
import json
import os
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset
from collections import defaultdict

# ================= 配置区域 =================
class Config:
    # 数据集路径
    DATA_ROOT = "./augmented_data_512"
    METADATA_FILE = "./augmented_data_512/metadata/test_samples.csv"  # 使用测试集
    SPLIT = "test"
    MAX_SAMPLES = None
    
    # 模型路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SD_LOCAL_PATH = os.path.join(BASE_DIR, "stable-diffusion-v1-5")
    
    # 微调后的 ControlNet 路径 (指向 checkpoint 目录下的 controlnet 文件夹)
    # 请根据实际情况修改 checkpoint 编号
    FINETUNED_CONTROLNET_PATH = "./final_model_output_viz_v4/checkpoint-50/controlnet"
    
    # Baseline ControlNet (原始 Canny)
    # 如果本地有 ControlNet-v1-1 且是 diffusers 格式，可指向它；否则使用 hub id
    BASELINE_CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
    
    OUTPUT_DIR = "./evaluation_results_comparison"
    BATCH_SIZE = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    SAVE_FIRST_N = 20
    NUM_INFERENCE_STEPS = 20
    # 展示时是否对生成图进行颜色匹配（仅用于 display，不用于指标计算）
    COLOR_MATCH_FOR_DISPLAY = True
    # 是否将匹配后的图用于指标计算（不推荐）
    APPLY_MATCH_TO_METRICS = False
    # 需要单独评估与保存的任务类别
    TASK_CATEGORIES = ["move_object", "drop_object", "cover_object"]

# ================= 数据集定义 (复用 trainNew.py 逻辑) =================
class SimpleVideoDataset(Dataset):
    def __init__(self, data_root, tokenizer, split='test', metadata_file=None, max_samples=None):
        import csv
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.split = split
        self.max_samples = max_samples
        
        # 确定 metadata 文件路径
        if metadata_file:
            meta_path = metadata_file
        else:
            meta_path = os.path.join(data_root, 'metadata', f'{split}_samples.csv')

        self.rows = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    self.rows.append(r)
        else:
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        if self.max_samples is not None:
            self.rows = self.rows[:self.max_samples]

    def __len__(self):
        return len(self.rows)

    def _load_npy(self, path):
        full = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        arr = np.load(full)
        return arr

    def _make_canny(self, frame):
        if frame.dtype != np.uint8:
            frame_u8 = (frame * 255).astype(np.uint8)
        else:
            frame_u8 = frame
        gray = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY)
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edge = cv2.Canny(gray, lower, upper)
        edge = np.stack([edge] * 3, axis=-1).astype(np.uint8)
        return Image.fromarray(edge)

    def _to_uint8_image(self, arr):
        if isinstance(arr, np.ndarray):
            if arr.dtype == np.uint8:
                return arr
            amin = float(np.min(arr))
            amax = float(np.max(arr))
            if amax <= 1.0 + 1e-6:
                img = (arr * 255.0).round().astype(np.uint8)
                return img
            img = np.clip(arr, 0, 255).round().astype(np.uint8)
            return img
        else:
            return np.array(arr).astype(np.uint8)

    def __getitem__(self, idx):
        row = self.rows[idx]
        inp_path = row['input_frames_path']
        tgt_path = row['target_frame_path']

        inp = self._load_npy(inp_path)
        tgt = self._load_npy(tgt_path)

        last = inp[-1]
        
        # 原始第20帧
        last_rgb = self._to_uint8_image(last)
        last_pil = Image.fromarray(last_rgb).convert("RGB")

        # 准备条件图 (PIL Image)
        cond_pil = self._make_canny(last)
        
        # 准备 GT 图 (PIL Image)
        tgt_rgb = self._to_uint8_image(tgt)
        tgt_pil = Image.fromarray(tgt_rgb).convert("RGB")

        prompt = row.get('label', row.get('template', 'interaction'))
        
        return {
            'conditioning_image': cond_pil,
            'ground_truth_image': tgt_pil,
            'last_frame_image': last_pil,
            'prompt': prompt,
            'video_id': row.get('video_id', str(idx)),
            'category': row.get('category', 'unknown')
        }

# ================= 指标计算 =================
def calculate_metrics(img1, img2):
    """计算 PSNR 和 SSIM"""
    # img1, img2: PIL Images or numpy arrays (H,W,C) RGB
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
        
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        score_psnr = psnr(img1, img2, data_range=255)
        score_ssim = ssim(img1, img2, channel_axis=2, data_range=255, win_size=3)
        return score_psnr, score_ssim
    except ImportError:
        print("⚠️  skimage不可用，使用OpenCV计算PSNR，SSIM设为0")
        score_psnr = cv2.PSNR(img1, img2)
        return score_psnr, 0.0

# ================= 主流程 =================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载基础组件 (VAE, UNet, TextEncoder, Tokenizer)
    print("📦 加载 Stable Diffusion 基础组件...")
    use_local = os.path.exists(Config.SD_LOCAL_PATH)
    model_path = Config.SD_LOCAL_PATH if use_local else "runwayml/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=use_local)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", local_files_only=use_local)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", local_files_only=use_local)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", local_files_only=use_local)
    
    # 2. 加载两个 ControlNet
    print(f"📦 加载 ControlNets...")
    # Baseline
    cnet_baseline = None
    if Config.BASELINE_CONTROLNET_PATH:
        print(f"   - Baseline: {Config.BASELINE_CONTROLNET_PATH}")
        try:
            baseline_is_local = os.path.exists(Config.BASELINE_CONTROLNET_PATH)
            cnet_baseline = ControlNetModel.from_pretrained(
                Config.BASELINE_CONTROLNET_PATH,
                local_files_only=baseline_is_local
            )
        except Exception as e:
            print(f"   ⚠️ 无法加载 Baseline ControlNet: {e}")

    # Finetuned
    cnet_finetuned = None
    if Config.FINETUNED_CONTROLNET_PATH:
        print(f"   - Finetuned: {Config.FINETUNED_CONTROLNET_PATH}")
        try:
            cnet_finetuned = ControlNetModel.from_pretrained(
                Config.FINETUNED_CONTROLNET_PATH,
                local_files_only=os.path.exists(Config.FINETUNED_CONTROLNET_PATH)
            )
        except Exception as e:
            print(f"   ⚠️ 无法加载 Finetuned ControlNet: {e}")
        
    if cnet_baseline is None and cnet_finetuned is None:
        print("❌ 两个模型都无法加载，退出。")
        return

    # 3. 准备 Pipelines
    def get_pipeline(controlnet):
        if controlnet is None: return None
        pipe = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler", local_files_only=use_local),
            safety_checker=None,
            feature_extractor=None
        )
        pipe.to(Config.DEVICE)
        if Config.DEVICE == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
        return pipe

    pipe_baseline = get_pipeline(cnet_baseline)
    pipe_finetuned = get_pipeline(cnet_finetuned)

    # 4. 数据集
    print("📂 准备数据集...")
    dataset = SimpleVideoDataset(
        Config.DATA_ROOT,
        tokenizer,
        split=Config.SPLIT,
        metadata_file=Config.METADATA_FILE,
        max_samples=Config.MAX_SAMPLES
    )
    print(f"   评估 split: {Config.SPLIT} | 样本数: {len(dataset)}")

    # 准备任务类别输出目录
    task_categories = Config.TASK_CATEGORIES
    if not task_categories:
        task_categories = sorted({row.get('category', 'unknown') for row in getattr(dataset, 'rows', [])})
    category_dirs = {}
    for cat in task_categories:
        cat_dir = os.path.join(Config.OUTPUT_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        category_dirs[cat] = cat_dir

    # 5. 评估循环
    results = []
    category_results = defaultdict(list)
    panel_counts = defaultdict(int)
    
    print("🚀 开始评估...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        prompt = item['prompt']
        cond_img = item['conditioning_image']
        gt_img = item['ground_truth_image']
        vid_id = item['video_id']
        category = item.get('category', 'unknown')
        sample_seed = Config.SEED + i
        if category not in category_dirs:
            cat_dir = os.path.join(Config.OUTPUT_DIR, category)
            os.makedirs(cat_dir, exist_ok=True)
            category_dirs[category] = cat_dir
        category_dir = category_dirs[category]
        
        # 生成 Baseline
        metrics_base = {"psnr": 0, "ssim": 0}
        baseline_image = None
        if pipe_baseline:
            gen_device = Config.DEVICE if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(sample_seed)
            out_base = pipe_baseline(
                prompt,
                image=cond_img,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                generator=generator
            ).images[0]
            out_base = out_base.resize(gt_img.size)
            p, s = calculate_metrics(gt_img, out_base)
            metrics_base = {"psnr": p, "ssim": s}
            baseline_image = out_base

        # 生成 Finetuned
        metrics_fine = {"psnr": 0, "ssim": 0}
        finetuned_image = None
        if pipe_finetuned:
            gen_device = Config.DEVICE if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(sample_seed)
            out_fine = pipe_finetuned(
                prompt,
                image=cond_img,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                generator=generator
            ).images[0]
            out_fine = out_fine.resize(gt_img.size)
            p, s = calculate_metrics(gt_img, out_fine)
            metrics_fine = {"psnr": p, "ssim": s}
            finetuned_image = out_fine

        # 保存单张四联图（第20帧、GT(第25帧)、baseline预测、finetuned预测）
        if panel_counts[category] < Config.SAVE_FIRST_N:
            base_size = gt_img.size
            last_vis = item.get('last_frame_image')
            if last_vis is None:
                last_vis = Image.new("RGB", base_size, (128, 128, 128))
            else:
                last_vis = last_vis.convert("RGB")
            last_vis = last_vis.resize(base_size, resample=Image.LANCZOS)

            gt_vis = gt_img.convert("RGB").resize(base_size, resample=Image.LANCZOS)

            gray_placeholder = Image.new("RGB", base_size, (128, 128, 128))

            if baseline_image is None:
                base_disp = gray_placeholder
            else:
                base_disp = baseline_image.convert("RGB").resize(base_size, resample=Image.LANCZOS)

            if finetuned_image is None:
                fine_disp = gray_placeholder
            else:
                fine_disp = finetuned_image.convert("RGB").resize(base_size, resample=Image.LANCZOS)

            if Config.COLOR_MATCH_FOR_DISPLAY:
                try:
                    from skimage.exposure import match_histograms
                    gt_arr = np.array(gt_vis)
                    if base_disp is not None and base_disp is not gray_placeholder:
                        b_arr = np.array(base_disp)
                        b_mat = match_histograms(b_arr, gt_arr, channel_axis=2)
                        base_disp = Image.fromarray(b_mat.astype('uint8'))
                    if fine_disp is not None and fine_disp is not gray_placeholder:
                        f_arr = np.array(fine_disp)
                        f_mat = match_histograms(f_arr, gt_arr, channel_axis=2)
                        fine_disp = Image.fromarray(f_mat.astype('uint8'))
                except Exception:
                    def match_mean_std(src_img, ref_img):
                        s = np.array(src_img).astype(np.float32)
                        r = np.array(ref_img).astype(np.float32)
                        for c in range(3):
                            s_mean, s_std = s[..., c].mean(), s[..., c].std()
                            r_mean, r_std = r[..., c].mean(), r[..., c].std()
                            if s_std > 1e-6:
                                s[..., c] = (s[..., c] - s_mean) / (s_std + 1e-6) * (r_std + 1e-6) + r_mean
                        s = np.clip(s, 0, 255).astype('uint8')
                        return Image.fromarray(s)

                    if base_disp is not None and base_disp is not gray_placeholder:
                        base_disp = match_mean_std(base_disp, gt_vis)
                    if fine_disp is not None and fine_disp is not gray_placeholder:
                        fine_disp = match_mean_std(fine_disp, gt_vis)

            panel = Image.new("RGB", (base_size[0] * 4, base_size[1]))
            panel.paste(last_vis, (0, 0))
            panel.paste(gt_vis, (base_size[0], 0))
            panel.paste(base_disp, (base_size[0] * 2, 0))
            panel.paste(fine_disp, (base_size[0] * 3, 0))

            panel.save(os.path.join(category_dir, f"{vid_id}_panel.jpg"))
            panel_counts[category] += 1

        result_entry = {
            "video_id": vid_id,
            "category": category,
            "baseline_psnr": metrics_base["psnr"],
            "baseline_ssim": metrics_base["ssim"],
            "finetuned_psnr": metrics_fine["psnr"],
            "finetuned_ssim": metrics_fine["ssim"]
        }
        results.append(result_entry)
        category_results[category].append(result_entry)

    # 6. 统计与保存
    df = pd.DataFrame(results)
    df["psnr_gain"] = df["finetuned_psnr"] - df["baseline_psnr"]
    df["ssim_gain"] = df["finetuned_ssim"] - df["baseline_ssim"]
    metrics_csv = os.path.join(Config.OUTPUT_DIR, "metrics_comparison.csv")
    df.to_csv(metrics_csv, index=False)
    
    print("\n📊 评估结果摘要:")
    print("-" * 40)
    summary = {}
    if pipe_baseline:
        summary["baseline_psnr"] = float(df['baseline_psnr'].mean())
        summary["baseline_ssim"] = float(df['baseline_ssim'].mean())
        print(f"Baseline  - PSNR: {summary['baseline_psnr']:.4f}, SSIM: {summary['baseline_ssim']:.4f}")
    if pipe_finetuned:
        summary["finetuned_psnr"] = float(df['finetuned_psnr'].mean())
        summary["finetuned_ssim"] = float(df['finetuned_ssim'].mean())
        print(f"Finetuned - PSNR: {summary['finetuned_psnr']:.4f}, SSIM: {summary['finetuned_ssim']:.4f}")
    if pipe_baseline and pipe_finetuned:
        summary["psnr_gain"] = float(df['psnr_gain'].mean())
        summary["ssim_gain"] = float(df['ssim_gain'].mean())
        print(f"提升     - ΔPSNR: {summary['psnr_gain']:.4f}, ΔSSIM: {summary['ssim_gain']:.4f}")
    print("-" * 40)
    summary_path = os.path.join(Config.OUTPUT_DIR, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存至: {metrics_csv}")
    print(f"摘要保存至: {summary_path}")

    # 逐任务统计与保存
    if category_results:
        print("\n📊 分任务评估:")
    for category in sorted(category_results.keys()):
        df_cat = df[df['category'] == category].copy()
        if df_cat.empty:
            continue
        cat_dir = category_dirs.get(category, os.path.join(Config.OUTPUT_DIR, category))
        os.makedirs(cat_dir, exist_ok=True)
        cat_csv = os.path.join(cat_dir, "metrics_comparison.csv")
        df_cat.to_csv(cat_csv, index=False)

        summary_cat = {}
        if pipe_baseline:
            summary_cat["baseline_psnr"] = float(df_cat['baseline_psnr'].mean())
            summary_cat["baseline_ssim"] = float(df_cat['baseline_ssim'].mean())
        if pipe_finetuned:
            summary_cat["finetuned_psnr"] = float(df_cat['finetuned_psnr'].mean())
            summary_cat["finetuned_ssim"] = float(df_cat['finetuned_ssim'].mean())
        if pipe_baseline and pipe_finetuned:
            summary_cat["psnr_gain"] = float(df_cat['psnr_gain'].mean())
            summary_cat["ssim_gain"] = float(df_cat['ssim_gain'].mean())

        summary_cat_path = os.path.join(cat_dir, "metrics_summary.json")
        with open(summary_cat_path, "w", encoding="utf-8") as f:
            json.dump(summary_cat, f, indent=2, ensure_ascii=False)

        print(f"[{category}] 样本数: {len(df_cat)}")
        if pipe_baseline:
            print(f"  Baseline  - PSNR: {summary_cat.get('baseline_psnr', 0):.4f}, SSIM: {summary_cat.get('baseline_ssim', 0):.4f}")
        if pipe_finetuned:
            print(f"  Finetuned - PSNR: {summary_cat.get('finetuned_psnr', 0):.4f}, SSIM: {summary_cat.get('finetuned_ssim', 0):.4f}")
        if pipe_baseline and pipe_finetuned:
            print(f"  提升      - ΔPSNR: {summary_cat.get('psnr_gain', 0):.4f}, ΔSSIM: {summary_cat.get('ssim_gain', 0):.4f}")

if __name__ == "__main__":
    main()
