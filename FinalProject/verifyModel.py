#!/usr/bin/env python3
"""
模型安装验证脚本
检查ControlNet和Stable Diffusion模型完整性
"""

import os
import sys
import torch
import yaml
from pathlib import Path
import json
import hashlib

class ModelVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_paths()
        
    def setup_paths(self):
        """设置模型路径"""
        self.paths = {
            # ControlNet 路径
            'controlnet_root': self.project_root / "ControlNet",
            'controlnet_models': self.project_root / "ControlNet/models",
            'controlnet_annotator': self.project_root / "ControlNet/annotator/ckpts",
            
            # Stable Diffusion 路径
            'sd_root': self.project_root / "stable-diffusion-v1-5",
            'sd_components': {
                'safety_checker': self.project_root / "stable-diffusion-v1-5/safety_checker",
                'text_encoder': self.project_root / "stable-diffusion-v1-5/text_encoder", 
                'unet': self.project_root / "stable-diffusion-v1-5/unet",
                'vae': self.project_root / "stable-diffusion-v1-5/vae"
            }
        }
        
    def check_directory_structure(self):
        """检查目录结构"""
        print("📁 检查目录结构...")
        
        issues = []
        
        # 检查ControlNet目录
        if not self.paths['controlnet_root'].exists():
            issues.append("❌ ControlNet根目录不存在")
        else:
            print("✅ ControlNet根目录存在")
            
        # 检查ControlNet模型目录
        if not self.paths['controlnet_models'].exists():
            issues.append("❌ ControlNet模型目录不存在")
        else:
            print("✅ ControlNet模型目录存在")
            
        # 检查Stable Diffusion目录
        if not self.paths['sd_root'].exists():
            issues.append("❌ Stable Diffusion根目录不存在")
        else:
            print("✅ Stable Diffusion根目录存在")
            
        # 检查SD组件目录
        for name, path in self.paths['sd_components'].items():
            if not path.exists():
                issues.append(f"❌ Stable Diffusion {name} 目录不存在")
            else:
                print(f"✅ Stable Diffusion {name} 目录存在")
                
        return len(issues) == 0, issues
    
    def check_controlnet_files(self):
        """检查ControlNet模型文件"""
        print("\n🔍 检查ControlNet模型文件...")
        
        required_files = [
            "control_sd15_canny.pth",
            "control_sd15_depth.pth", 
            "control_sd15_hed.pth",
            "control_sd15_mlsd.pth",
            "control_sd15_normal.pth",
            "control_sd15_openpose.pth",
            "control_sd15_scribble.pth", 
            "control_sd15_seg.pth"
        ]
        
        config_files = ["cldm_v15.yaml"]
        
        found_files = []
        missing_files = []
        
        # 检查模型文件
        for file in required_files:
            file_path = self.paths['controlnet_models'] / file
            if file_path.exists():
                size = file_path.stat().st_size / (1024**2)  # MB
                print(f"✅ {file}: {size:.1f} MB")
                found_files.append(file)
            else:
                print(f"❌ {file}: 缺失")
                missing_files.append(file)
                
        # 检查配置文件
        for file in config_files:
            file_path = self.paths['controlnet_models'] / file
            if file_path.exists():
                print(f"✅ {file}: 存在")
                found_files.append(file)
            else:
                print(f"❌ {file}: 缺失")
                missing_files.append(file)
                
        return len(missing_files) == 0, found_files, missing_files
    
    def check_sd_files(self):
        """检查Stable Diffusion文件"""
        print("\n🔍 检查Stable Diffusion文件...")
        
        required_files = {
            'safety_checker': ['config.json', 'model.safetensors'],
            'text_encoder': ['config.json', 'model.safetensors'],
            'unet': ['config.json', 'diffusion_pytorch_model.safetensors'],
            'vae': ['config.json', 'diffusion_pytorch_model.safetensors']
        }
        
        found_files = []
        missing_files = []
        
        for component, files in required_files.items():
            component_path = self.paths['sd_components'][component]
            
            if not component_path.exists():
                for file in files:
                    missing_files.append(f"{component}/{file}")
                continue
                    
            for file in files:
                file_path = component_path / file
                if file_path.exists():
                    size = file_path.stat().st_size / (1024**2)  # MB
                    print(f"✅ {component}/{file}: {size:.1f} MB")
                    found_files.append(f"{component}/{file}")
                else:
                    print(f"❌ {component}/{file}: 缺失")
                    missing_files.append(f"{component}/{file}")
                    
        return len(missing_files) == 0, found_files, missing_files
    
    def check_annotator_files(self):
        """检查注释器文件"""
        print("\n🔍 检查注释器文件...")
        
        required_files = [
            "body_pose_model.pth",
            "dpt_hybrid-midas-501f0c75.pt", 
            "hand_pose_model.pth",
            "mlsd_large_512_fp32.pth",
            "network-bsds500.pth",
            "upernet_global_small.pth"
        ]
        
        found_files = []
        missing_files = []
        
        for file in required_files:
            file_path = self.paths['controlnet_annotator'] / file
            if file_path.exists():
                size = file_path.stat().st_size / (1024**2)  # MB
                print(f"✅ {file}: {size:.1f} MB")
                found_files.append(file)
            else:
                print(f"⚠️  {file}: 缺失 (可选)")
                # 注释器文件是可选的，所以不标记为严重错误
                
        return len(found_files) > 0, found_files, missing_files
    
    def test_model_loading(self):
        """测试模型加载能力"""
        print("\n🚀 测试模型加载...")
        
        try:
            # 测试PyTorch基础功能
            print("测试PyTorch...")
            print(f"  PyTorch版本: {torch.__version__}")
            print(f"  CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # 测试模型文件完整性
            print("\n测试模型文件完整性...")
            
            # 测试ControlNet模型文件
            canny_path = self.paths['controlnet_models'] / "control_sd15_canny.pth"
            if canny_path.exists():
                try:
                    state_dict = torch.load(canny_path, map_location='cpu')
                    print(f"✅ ControlNet模型可加载，参数数量: {len(state_dict)}")
                except Exception as e:
                    print(f"❌ ControlNet模型加载失败: {e}")
                    return False
            
            # 测试Stable Diffusion组件
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
                from diffusers import AutoencoderKL, UNet2DConditionModel
                
                # 测试文本编码器
                text_encoder = CLIPTextModel.from_pretrained(
                    self.paths['sd_components']['text_encoder']
                )
                print("✅ 文本编码器加载成功")
                
                # 测试VAE
                vae = AutoencoderKL.from_pretrained(
                    self.paths['sd_components']['vae'] 
                )
                print("✅ VAE加载成功")
                
                # 测试UNet
                unet = UNet2DConditionModel.from_pretrained(
                    self.paths['sd_components']['unet']
                )
                print("✅ UNet加载成功")
                
            except Exception as e:
                print(f"❌ Stable Diffusion组件加载失败: {e}")
                return False
                
            print("✅ 所有模型加载测试通过!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载测试失败: {e}")
            return False
    
    def check_dependencies(self):
        """检查必要的依赖"""
        print("\n📦 检查依赖...")
        
        dependencies = [
            'torch', 'torchvision', 'numpy', 'PIL', 'opencv-python',
            'transformers', 'diffusers', 'accelerate', 'safetensors',
            'omegaconf', 'einops', 'xformers'
        ]
        
        missing_deps = []
        for dep in dependencies:
            try:
                if dep == 'PIL':
                    import PIL
                    version = PIL.__version__
                elif dep == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                else:
                    module = __import__(dep)
                    version = getattr(module, '__version__', '未知')
                print(f"✅ {dep}: {version}")
            except ImportError:
                print(f"❌ {dep}: 未安装")
                missing_deps.append(dep)
                
        if missing_deps:
            print(f"\n⚠️  缺少依赖，安装命令:")
            print(f"pip install {' '.join(missing_deps)}")
            return False
        return True
    
    def create_project_structure(self):
        """创建标准项目结构"""
        print("\n📁 创建标准项目结构...")
        
        directories = [
            "checkpoints",
            "results/training_results",
            "results/generated_frames", 
            "results/evaluation",
            "logs",
            "configs",
            "scripts",
            "data/raw",
            "data/processed"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  创建: {dir_path}")
            
        # 创建符号链接到标准位置
        self.create_symbolic_links()
        
        print("✅ 项目结构创建完成")
    
    def create_symbolic_links(self):
        """创建符号链接"""
        try:
            # 链接ControlNet模型到标准位置
            models_target = self.project_root / "models"
            models_target.mkdir(exist_ok=True)
            
            controlnet_link = models_target / "controlnet"
            if not controlnet_link.exists():
                os.symlink(self.paths['controlnet_models'], controlnet_link)
                print(f"🔗 创建符号链接: models/controlnet -> ControlNet/models")
                
            # 链接Stable Diffusion到标准位置  
            sd_link = models_target / "stable-diffusion"
            if not sd_link.exists():
                os.symlink(self.paths['sd_root'], sd_link)
                print(f"🔗 创建符号链接: models/stable-diffusion -> stable-diffusion-v1-5")
                
        except Exception as e:
            print(f"⚠️  创建符号链接失败: {e}")
            print("  将使用原始路径")
    
    def save_verification_report(self, results):
        """保存验证报告"""
        report = {
            "verification_date": str(torch.datetime.now()) if hasattr(torch, 'datetime') else "unknown",
            "system_info": {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
            },
            "results": results
        }
        
        report_path = self.project_root / "model_verification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📊 验证报告已保存: {report_path}")
    
    def run_full_verification(self):
        """运行完整验证"""
        print("=" * 60)
        print("🔍 开始模型完整性验证")
        print("=" * 60)
        
        results = {
            'directory_structure': False,
            'controlnet_files': False, 
            'sd_files': False,
            'annotator_files': False,
            'dependencies': False,
            'model_loading': False
        }
        
        issues = []
        
        # 1. 检查目录结构
        results['directory_structure'], dir_issues = self.check_directory_structure()
        issues.extend(dir_issues)
        
        # 2. 检查ControlNet文件
        results['controlnet_files'], found_ctrl, missing_ctrl = self.check_controlnet_files()
        if not results['controlnet_files']:
            issues.append(f"ControlNet文件缺失: {missing_ctrl}")
        
        # 3. 检查Stable Diffusion文件  
        results['sd_files'], found_sd, missing_sd = self.check_sd_files()
        if not results['sd_files']:
            issues.append(f"Stable Diffusion文件缺失: {missing_sd}")
            
        # 4. 检查注释器文件
        results['annotator_files'], found_ann, missing_ann = self.check_annotator_files()
        
        # 5. 检查依赖
        results['dependencies'] = self.check_dependencies()
        if not results['dependencies']:
            issues.append("缺少必要的依赖包")
            
        # 6. 测试模型加载（只在其他检查通过时进行）
        if all([results['directory_structure'], results['controlnet_files'], results['sd_files']]):
            results['model_loading'] = self.test_model_loading()
            if not results['model_loading']:
                issues.append("模型加载测试失败")
        else:
            print("\n⚠️  跳过模型加载测试（基础检查未通过）")
            results['model_loading'] = False
            
        # 创建项目结构
        self.create_project_structure()
        
        # 保存报告
        self.save_verification_report(results)
        
        # 输出总结
        print("\n" + "=" * 60)
        print("验证总结")
        print("=" * 60)
        
        success_count = sum(results.values())
        total_count = len(results)
        
        if success_count == total_count:
            print("🎉 所有验证通过！项目准备就绪。")
            return True, issues
        else:
            print(f"⚠️  {success_count}/{total_count} 项验证通过")
            if issues:
                print("\n需要解决的问题:")
                for issue in issues:
                    print(f"  - {issue}")
            return False, issues

def main():
    """主函数"""
    verifier = ModelVerifier()
    success, issues = verifier.run_full_verification()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 下一步行动指南")
        print("=" * 60)
        print("1. 📊 验证数据加载器:")
        print("   python -c \"from dataloader import test_data_loader; test_data_loader()\"")
        print("")
        print("2. 🚀 开始模型训练:")
        print("   python scripts/train_frame_prediction.py")
        print("")
        print("3. 📈 监控训练进度:")
        print("   tail -f logs/training.log")
        print("")
        print("4. 🧪 训练完成后进行推理测试:")
        print("   python scripts/inference.py")
        print("")
        print("5. 📊 评估模型性能:")
        print("   python scripts/evaluate.py")
        print("")
        print("💡 提示: 你的数据加载器已经准备好，可以直接开始训练!")
    else:
        print("❌ 验证失败")
        print("=" * 60)
        print("请先解决上述问题，然后重新运行验证脚本。")
        sys.exit(1)

if __name__ == "__main__":
    main()