#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
import fnmatch
import zipfile

# 这个库手动下载一下：git clone https://github.com/FunAudioLLM/CosyVoice.git
def check_installation(packages=None):
    """检查必要的包是否已安装"""
    if packages is None:
        packages = ['modelscope', 'huggingface_hub']
    missing_packages = []

    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")

    return missing_packages


def install_missing_packages(missing_packages):
    """安装缺失的包"""
    if missing_packages:
        print(f"安装缺失的包: {missing_packages}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    else:
        print("所有必要的包都已安装")


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def download_cosyvoice(prefer: str = 'modelscope', target_dir: str = 'CosyVoice-300M'):
    """下载 CosyVoice-300M 模型到指定目录，并按照 prefer 顺序尝试。

    prefer: 'modelscope' 或 'hf'
    target_dir: 期望的本地保存目录
    """
    print("开始下载 CosyVoice-300M 模型...")
    _ensure_parent_dir(target_dir)

    errors = []

    def try_modelscope():
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download('iic/CosyVoice-300M', local_dir=target_dir)
            print(f"✓ 从 ModelScope 下载完成: {model_dir}")
            return model_dir
        except Exception as e:
            msg = f"ModelScope 下载失败: {e}"
            print(msg)
            errors.append(msg)
            return None

    def try_hf():
        try:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(
                'FunAudioLLM/CosyVoice-300M',
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ 从 Hugging Face 下载完成: {model_dir}")
            return model_dir
        except Exception as e:
            msg = f"Hugging Face 下载失败: {e}"
            print(msg)
            errors.append(msg)
            return None

    # 根据 prefer 顺序尝试
    order = ['modelscope', 'hf'] if prefer == 'modelscope' else ['hf', 'modelscope']
    model_dir = None
    for src in order:
        if src == 'modelscope':
            model_dir = try_modelscope()
        else:
            model_dir = try_hf()
        if model_dir:
            break

    if not model_dir:
        print("❌ 所有下载方法都失败了，请手动下载")
        for e in errors:
            print(f"  - {e}")
        return None

    return model_dir


def _glob_any(root: str, patterns):
    """在 root 下递归匹配任意一个 pattern，返回匹配到的文件列表"""
    matched = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            for p in patterns:
                if fnmatch.fnmatch(name, p):
                    matched.append(os.path.join(dirpath, name))
                    break
    return matched


def _extract_archives(model_dir: str):
    """解压目录下的 zip 权重包（如果尚未解压）。"""
    zip_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    if not zip_files:
        return
    for z in zip_files:
        zip_path = os.path.join(model_dir, z)
        extract_dir = os.path.join(model_dir, z.rsplit('.zip', 1)[0])
        # 若目录已存在且有内容则跳过
        if os.path.isdir(extract_dir) and os.listdir(extract_dir):
            print(f"跳过已解压: {z}")
            continue
        try:
            print(f"解压 {z} -> {extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"⚠ 解压失败 {z}: {e}")


def verify_download(model_dir: str, auto_extract: bool = False):
    """验证下载是否完整。

    基础校验：
    - 至少存在一个权重文件：*.safetensors 或 pytorch_model*.bin
    - 存在至少一个 config.json（任意子目录）
    """
    if not model_dir or not os.path.isdir(model_dir):
        print("✗ 模型目录不存在")
        return False

    print("验证下载的文件...")

    if auto_extract:
        _extract_archives(model_dir)

    # 权重文件：接受 safetensors/bin 以及发布的 .pt / .onnx 文件
    weight_files = _glob_any(model_dir, ["*.safetensors", "pytorch_model*.bin", "model*.bin", "*.pt", "*.onnx"])
    # 配置文件：CosyVoice 发布中使用 configuration.json 或 cosyvoice.yaml
    config_files = _glob_any(model_dir, ["config.json", "configuration.json", "cosyvoice.yaml"])

    if weight_files:
        print(f"✓ 权重文件数量: {len(weight_files)}（示例: {os.path.basename(weight_files[0])}）")
    else:
        print("✗ 未找到权重文件（*.safetensors 或 pytorch_model*.bin）")

    if config_files:
        print(f"✓ 配置文件数量: {len(config_files)}")
    else:
        print("✗ 未找到 config.json")

    basic_ok = bool(weight_files and config_files)
    if not basic_ok:
        return False

    print("✓ 基础校验通过")
    return True


def main():
    parser = argparse.ArgumentParser(description="CosyVoice-300M 安装与校验")
    parser.add_argument(
        "--target-dir",
        default="CosyVoice-300M",
        help="模型下载保存目录（默认: CosyVoice-300M，位于项目根目录）",
    )
    parser.add_argument(
        "--prefer",
        choices=["modelscope", "hf"],
        default="modelscope",
        help="优先下载源（默认: modelscope）",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="强制重新下载，即使目录已存在",
    )
    args = parser.parse_args()

    print("开始设置 CosyVoice 环境...")

    # 1. 检查安装（下载器依赖）
    missing_packages = check_installation(['modelscope', 'huggingface_hub'])

    # 2. 安装缺失的包
    install_missing_packages(missing_packages)

    # 3. 下载模型（如果目录不存在或为空则执行，或强制下载）
    need_download = (not os.path.isdir(args.target_dir)) or \
                   (os.path.isdir(args.target_dir) and not os.listdir(args.target_dir)) or \
                   args.force_download
                   
    if need_download:
        if args.force_download and os.path.exists(args.target_dir):
            print(f"强制重新下载，删除现有目录: {args.target_dir}")
            import shutil
            shutil.rmtree(args.target_dir)
            
        model_dir = download_cosyvoice(prefer=args.prefer, target_dir=args.target_dir)
    else:
        print(f"目录 {args.target_dir} 已存在且非空，跳过下载。")
        model_dir = args.target_dir

    # 4. 验证下载
    if model_dir and verify_download(model_dir, auto_extract=True):
        print("\n🎉 CosyVoice 设置完成！")
        print(f"模型路径: {model_dir}")
        
        # 验证关键文件
        onnx_path = os.path.join(model_dir, "speech_tokenizer_v1.onnx")
        if os.path.exists(onnx_path):
            print(f"✓ 关键文件验证: speech_tokenizer_v1.onnx 存在")
            
            # 输出可用路径信息
            print("\n✅ 配置完成！可以直接使用以下路径：")
            print(f"  - ONNX文件: {model_dir}/speech_tokenizer_v1.onnx")
            print(f"  - 模型目录: {model_dir}")
            
            # 提醒用户更新其他脚本中的路径
            print("\n💡 提示: 请确保其他脚本中的路径指向:")
            print(f"  - s3.sh 中的 ONNX_PATH 设置为: {model_dir}/speech_tokenizer_v1.onnx")
            print(f"  - utt2text_and_feature.py 中的 model_dir 设置为: {model_dir}")
            
        else:
            print(f"❌ 关键文件缺失: speech_tokenizer_v1.onnx")
            return 1
            
    else:
        print("\n❌ CosyVoice 设置失败")
        print("请尝试以下解决方案：")
        print("1) 手动下载:")
        print("   - ModelScope: https://www.modelscope.cn/models/iic/CosyVoice-300M")
        print("   - HuggingFace: https://huggingface.co/FunAudioLLM/CosyVoice-300M")
        print("2) 使用 --force-download 参数强制重新下载")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())