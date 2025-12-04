#!/usr/bin/env python3
"""
ControlNet Zero-Shot Baseline 预测脚本 (本地模型适配版)
用途：使用官方权重（不微调）直接预测，用于生成对比图。
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    # 只依赖 diffusers + 本地 loader；不依赖 cldm（如无 cldm 则直接使用 hub 或 diffusers 本地目录）
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from loaderData512 import create_task_specific_loaders  # ✅ 确保使用 512 数据加载器
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


class ZeroShotPredictor:
    def __init__(self, output_dir="results_zeroshot"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 Zero-Shot 预测模式 | Device: {self.device}")

        # ================= 配置路径 (请核对这里) =================
        # 1. SD v1.5 的本地路径 (和训练脚本保持一致)
        self.sd_model_path = str(project_root / "stable-diffusion-v1-5")
        if not os.path.exists(self.sd_model_path):
            # 兼容旧的路径写法
            alt = str(project_root / "stable-diffusion-v1_5")
            if os.path.exists(alt):
                self.sd_model_path = alt
            else:
                # 尝试默认 hub 名称作为回退
                self.sd_model_path = "runwayml/stable-diffusion-v1-5"

        # 2. ControlNet 本地目录/文件（优先使用项目中的 ControlNet 模型）
        # 我们会尝试几种本地位置：
        # - FinalProject/ControlNet/models/control_sd15_canny.pth
        # - FinalProject/ControlNet-v1-1/control_v11p_sd15_canny.pth (+ yaml)
        self.cn_model_dir = str(project_root / "ControlNet")
        self.cn_v11_dir = str(project_root / "ControlNet-v1-1")
        # =======================================================

        # 加载 ControlNet
        self.controlnet = self._load_controlnet()

        # 加载 SD Pipe
        print(f"📦 加载 Stable Diffusion 底座: {self.sd_model_path}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True if os.path.exists(self.sd_model_path) else False
        ).to(self.device)

        # 优化设置
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def _load_controlnet(self):
        """加载 ControlNet，优先使用本地转换好的 diffusers 格式，或者转换 .pth"""
        print("📦 正在加载官方 ControlNet Canny 权重...")
        # 如果仓库里只有 cldm/.pth 文件但没有 cldm 环境，无法直接加载；先检测并提示
        try:
            alt_pth = os.path.join(self.cn_model_dir, 'models', 'control_sd15_canny.pth')
            v11_pth = os.path.join(self.cn_v11_dir, 'control_v11p_sd15_canny.pth')
            if os.path.exists(alt_pth) or os.path.exists(v11_pth):
                print(f"⚠️ 发现本地 .pth 文件，但当前环境没有 cldm，无法直接用 .pth 加载: {alt_pth if os.path.exists(alt_pth) else v11_pth}")
                print("   建议：要么安装 cldm（可用 create_model/load_state_dict 转换），要么把模型转换为 diffusers 格式后放在本地目录。脚本将继续尝试从本地 diffusers 目录或 hub 加载。")
        except Exception:
            pass

        # 如果本地没有可直接加载的 cldm 文件，尝试用 diffusers 从本地目录（若存在）或 hub 加载
        try:
            if os.path.exists(self.cn_v11_dir):
                # 如果存在一个已转换的 diffusers 风格目录则直接加载
                try:
                    print(f"🔁 尝试从本地 diffusers 风格目录加载 ControlNet: {self.cn_v11_dir}")
                    return ControlNetModel.from_pretrained(self.cn_v11_dir, torch_dtype=torch.float16).to(self.device)
                except Exception:
                    pass

            if os.path.exists(self.cn_model_dir):
                try:
                    print(f"🔁 尝试从本地 diffusers 风格目录加载 ControlNet: {self.cn_model_dir}")
                    return ControlNetModel.from_pretrained(self.cn_model_dir, torch_dtype=torch.float16).to(self.device)
                except Exception:
                    pass

            # 最后回退：在线加载官方预训练的 sd-controlnet-canny
            print("🌐 回退到在线加载 lllyasviel/sd-controlnet-canny")
            return ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16,
                local_files_only=False
            ).to(self.device)
        except Exception as e:
            print(f"❌ 严重错误: 无法加载 Canny ControlNet 模型: {e}")
            raise

    def get_canny_image(self, tensor_img):
        # 转换逻辑保持不变，确保处理的是 512 图
        img = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if img.min() < 0:
            img = (img + 1) / 2
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 100, 200)
        edge = np.stack([edge] * 3, axis=-1)
        return Image.fromarray(edge), Image.fromarray(img)

    def run_prediction(self, dataloader, task_name, num_samples=20):
        # ... (和之前的逻辑完全一致，代码省略以节省篇幅，直接用上面的 run_prediction 即可) ...
        # 这里把 run_prediction 的代码完整粘贴过来
        print(f"\n🎨 开始任务 {task_name} 的 Zero-Shot 预测...")
        save_path = self.output_dir / task_name
        save_path.mkdir(exist_ok=True)

        count = 0
        for batch in dataloader:
            if count >= num_samples: break

            # 确保只取 Frame 20
            frame_20_batch = batch['input_frames'][:, -1]
            target_batch = batch['target_frame']
            prompts = batch.get('label_text', ['moving object'] * len(frame_20_batch))

            for i in range(len(frame_20_batch)):
                if count >= num_samples: break

                canny_pil, frame_20_pil = self.get_canny_image(frame_20_batch[i:i + 1])

                gt_np = target_batch[i].permute(1, 2, 0).cpu().numpy()
                if gt_np.min() < 0: gt_np = (gt_np + 1) / 2
                gt_pil = Image.fromarray((gt_np * 255).astype(np.uint8))

                seed = torch.Generator(device="cpu").manual_seed(42)

                with torch.inference_mode():
                    output_image = self.pipe(
                        prompt=prompts[i],
                        image=canny_pil,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        generator=seed
                    ).images[0]

                w, h = 512, 512
                grid = Image.new('RGB', (w * 4, h))
                grid.paste(frame_20_pil, (0, 0))
                grid.paste(canny_pil, (w, 0))
                grid.paste(output_image, (w * 2, 0))
                grid.paste(gt_pil, (w * 3, 0))

                file_name = f"sample_{count:03d}.jpg"
                grid.save(save_path / file_name)
                count += 1
                print(f"✅ 已保存: {task_name}/{file_name}")


def main():
    batch_size = 4
    num_samples_to_visualize = 20  # 跑 20 张来看看就行
    tasks = ['move_object', 'drop_object', 'cover_object']

    predictor = ZeroShotPredictor()

    for task in tasks:
        try:
            # ✅ 这里调用的是 loaderData512，保证加载的是 512 数据
            # 且加载的是 test_loader (测试集)
            _, _, test_loader = create_task_specific_loaders(
                task_name=task,
                batch_size=batch_size,
                data_path="processed_data_512"
            )

            if len(test_loader) == 0: continue
            predictor.run_prediction(test_loader, task, num_samples_to_visualize)

        except Exception as e:
            print(f"❌ 任务 {task} 出错: {e}")


if __name__ == "__main__":
    main()