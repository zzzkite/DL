#!/usr/bin/env python3
import sys
import subprocess
import platform

def run_command(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Exception: {str(e)}"

def get_system_info():
    """获取系统信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"平台: {platform.platform()}")
    print(f"处理器: {platform.processor()}")
    print(f"机器类型: {platform.machine()}")

def get_python_info():
    """获取Python信息"""
    print("\n" + "=" * 60)
    print("Python信息")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")

def get_pytorch_info():
    """获取PyTorch信息"""
    print("\n" + "=" * 60)
    print("PyTorch信息")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"Torchvision版本: {torch.version.__version__ if hasattr(torch.version, '__version__') else 'N/A'}")
        
        # CUDA信息
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA: 不可用")
            
        # 编译选项
        print(f"使用MKL: {torch.backends.mkl.is_available()}")
        print(f"使用CUDA DNN: {torch.backends.cudnn.enabled}")
        
    except ImportError as e:
        print(f"PyTorch导入错误: {e}")
    except Exception as e:
        print(f"获取PyTorch信息时出错: {e}")

def get_torchaudio_info():
    """获取torchaudio信息"""
    print("\n" + "=" * 60)
    print("torchaudio信息")
    print("=" * 60)
    try:
        import torchaudio
        print(f"torchaudio版本: {torchaudio.__version__}")
        # 检查torchaudio的后端
        print(f"torchaudio后端: {torchaudio._backend.get_audio_backend()}")
    except ImportError as e:
        print(f"torchaudio导入错误: {e}")
    except Exception as e:
        print(f"获取torchaudio信息时出错: {e}")

def get_other_libs():
    """获取其他重要库的信息"""
    print("\n" + "=" * 60)
    print("其他重要库信息")
    print("=" * 60)
    
    libs = [
        "transformers", "datasets", "accelerate", 
        "huggingface_hub", "modelscope", "tqdm", 
        "matplotlib", "librosa", "soundfile"
    ]
    
    for lib in libs:
        try:
            module = __import__(lib)
            if hasattr(module, '__version__'):
                print(f"{lib}: {module.__version__}")
            else:
                print(f"{lib}: 已安装 (版本信息不可用)")
        except ImportError:
            print(f"{lib}: 未安装")

def get_cuda_system_info():
    """获取系统CUDA信息"""
    print("\n" + "=" * 60)
    print("系统CUDA信息")
    print("=" * 60)
    
    # 检查nvcc
    nvcc_result = run_command("nvcc --version")
    if "release" in nvcc_result:
        for line in nvcc_result.split('\n'):
            if "release" in line:
                print(f"系统CUDA版本: {line}")
                break
    else:
        print("系统CUDA: nvcc未找到或不可用")
    
    # 检查nvidia-smi
    nvidia_result = run_command("nvidia-smi")
    if "CUDA Version" in nvidia_result:
        for line in nvidia_result.split('\n'):
            if "CUDA Version" in line:
                print(f"驱动CUDA版本: {line.strip()}")
                break
    else:
        print("NVIDIA驱动: nvidia-smi未找到或不可用")

def main():
    print("开始收集环境信息...\n")
    
    get_system_info()
    get_python_info()
    get_pytorch_info()
    get_torchaudio_info()
    get_other_libs()
    get_cuda_system_info()
    
    print("\n" + "=" * 60)
    print("环境信息收集完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()