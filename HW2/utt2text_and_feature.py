#!/usr/bin/env python3
import sys
import os
import gc
import psutil  # 添加系统监控

# 添加 CosyVoice 到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
cosyvoice_path = os.path.join(current_dir, 'CosyVoice')

# 添加 CosyVoice 主目录
if cosyvoice_path not in sys.path:
    sys.path.insert(0, cosyvoice_path)

# 添加 CosyVoice 的第三方依赖路径
matcha_path = os.path.join(cosyvoice_path, 'third_party', 'Matcha-TTS')
if os.path.exists(matcha_path) and matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

print(f"添加 CosyVoice 路径: {cosyvoice_path}")

import argparse
import json
import signal  # 添加信号处理
import glob

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# 全局变量，用于优雅退出
stop_processing = False

def signal_handler(sig, frame):
    global stop_processing
    print("\n收到中断信号，正在优雅退出...")
    stop_processing = True

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- huggingface_hub compatibility patch (for CosyVoice) ---
try:
    import huggingface_hub as _hfh
    if not hasattr(_hfh, "cached_download"):
        from huggingface_hub import hf_hub_download as _hf_hub_download

        def cached_download(*args, **kwargs):
            return _hf_hub_download(*args, **kwargs)

        _hfh.cached_download = cached_download
except Exception:
    pass

# 尝试导入 CosyVoice
try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    print("✅ CosyVoice 导入成功")
except ImportError as e:
    print(f"❌ CosyVoice 导入失败: {e}")
    print("请确保 CosyVoice 目录存在且包含必要的文件")
    sys.exit(1)


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_audio(path, target_sr=16000):
    try:
        audio, sr = torchaudio.load(path)
        if sr != target_sr:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        return audio.squeeze(0), target_sr  # (T,), sr
    except Exception as e:
        print(f"音频加载失败 {path}: {e}")
        return None, None


def extract_whisper_encoder_feats(waveform, model, processor, device, max_duration=30.0):
    if waveform is None:
        return None, None
        
    # waveform: 1D torch tensor at 16k
    num_seconds = waveform.numel() / 16000.0
    if num_seconds > max_duration:
        return None, None

    try:
        audio_np = waveform.numpy()
        inputs = processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            enc_out = model.model.encoder(
                input_features=input_features,
                output_hidden_states=True,
            )

        hidden_states = enc_out.hidden_states  # list: [layer0, layer1, ..., last]
        mid_idx = len(hidden_states) // 2
        mid_layer = hidden_states[mid_idx].cpu().squeeze(0)   # (T, D)
        final_layer = hidden_states[-1].cpu().squeeze(0)      # (T, D)
        
        # 清理中间变量
        del enc_out, hidden_states, inputs, input_features
        return mid_layer, final_layer
    except Exception as e:
        print(f"Whisper特征提取失败: {e}")
        return None, None


def check_system_resources():
    """检查系统资源使用情况"""
    memory = psutil.virtual_memory()
    print(f"内存使用: {memory.percent}% ({memory.used//1024//1024}MB / {memory.total//1024//1024}MB)")
    
    if memory.percent > 90.0:
        print(f"❌ 严重警告: 内存使用率超过 90% ({memory.percent}%)，为防止系统卡死，程序将自动终止。")
        print("建议: 请尝试减小 --subset_ratio 或使用分块保存版本的脚本。")
        # 尝试保存已处理的数据（尽力而为）
        global stop_processing
        stop_processing = True
        return False

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存使用: 已分配 {gpu_memory:.2f} GB, 保留 {gpu_reserved:.2f} GB")
    
    return True


def main(args):
    global stop_processing
    
    # 设置环境变量减少内存碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 加载 CosyVoice 用于文本嵌入
    print(f"加载 CosyVoice 模型从: {args.model_dir}")
    try:
        cosy = CosyVoice(args.model_dir)
        # 提取需要的 embedding 层并保持引用
        emb_layer = cosy.model.llm.text_embedding
        
        # 优化显存：删除 CosyVoice 不需要的大部分模块
        print("正在释放 CosyVoice 非必要模块以节省显存...")
        if hasattr(cosy.model, 'llm'):
            del cosy.model.llm
        if hasattr(cosy.model, 'flow'):
            del cosy.model.flow
        if hasattr(cosy.model, 'hift'):
            del cosy.model.hift
            
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("CosyVoice 显存优化完成")

    except Exception as e:
        print(f"CosyVoice 模型加载失败: {e}")
        return

    # 加载 Whisper 编码器 - 使用更小的模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    print("加载 Whisper 模型...")
    try:
        # 使用最小的模型
        processor = AutoProcessor.from_pretrained("openai/whisper-base")
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-base"
        ).to(device)
        whisper_model.eval()
    except Exception as e:
        print(f"Whisper 模型加载失败: {e}")
        return

    # 加载数据
    try:
        data = load_jsonl(args.jsonl)
        print(f"从 {args.jsonl} 加载了 {len(data)} 个项目")
        
        if args.subset_ratio < 1.0 and args.subset_ratio > 0.0:
            original_len = len(data)
            subset_size = int(original_len * args.subset_ratio)
            data = data[:subset_size]
            print(f"应用数据集比例 {args.subset_ratio}: 从 {original_len} 减少到 {len(data)} 个项目")
            
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 分批处理配置
    # BATCH_SIZE = 1000  # 不再使用分批保存到磁盘
    
    # --- 阶段1: 扫描已处理的数据 (仅获取Keys，不保留数据以节省显存) ---
    # 简化逻辑：如果文件存在，直接加载到内存中作为基础，继续追加
    # 如果内存不够，这是用户接受的风险（因为要求一次性生成）
    
    existing_text_emb = {}
    existing_whisper_mid = {}
    existing_whisper_final = {}
    
    if os.path.exists(args.output_text):
        try:
            print(f"加载已存在的文本嵌入: {args.output_text}")
            existing_text_emb = torch.load(args.output_text, map_location='cpu')
        except Exception as e:
            print(f"加载现有文本嵌入失败: {e}")

    if os.path.exists(args.output_whisper):
        try:
            print(f"加载已存在的Whisper特征: {args.output_whisper}")
            whisper_data = torch.load(args.output_whisper, map_location='cpu')
            existing_whisper_mid = whisper_data.get("mid", {})
            existing_whisper_final = whisper_data.get("final", {})
        except Exception as e:
            print(f"加载现有Whisper特征失败: {e}")

    processed_keys = set(existing_text_emb.keys()) & set(existing_whisper_mid.keys())
    print(f"总计已处理: {len(processed_keys)} 个项目")

    # 过滤待处理数据
    data_to_process = [item for item in data if item["audio_path"] not in processed_keys]
    print(f"剩余待处理: {len(data_to_process)} 个项目")

    processed_count = 0
    whisper_success_count = 0
    whisper_fail_count = 0

    print("开始处理数据...")
    
    # 初始内存清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 处理数据
    for i, item in enumerate(tqdm(data_to_process)):
        if stop_processing:
            print("检测到停止信号，提前退出...")
            break
            
        audio_path = item["audio_path"]
        text = item["text"]

        # 每处理5个样本强制清理一次内存
        if i % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 每处理20个样本检查一次系统资源
        if i % 20 == 0:
            if not check_system_resources():
                break

        # ----- CosyVoice 文本嵌入 -----
        try:
            text_token, text_token_len = cosy.frontend._extract_text_token(text)
            with torch.no_grad():
                text_token = text_token.to(emb_layer.weight.device).long()
                text_emb = emb_layer(text_token)
            existing_text_emb[audio_path] = text_emb.squeeze(0).cpu()
            
            # 立即释放内存
            del text_token, text_emb
            processed_count += 1
            
        except Exception as e:
            print(f"文本嵌入提取失败 {audio_path}: {e}")
            continue

        # ----- Whisper 编码器特征 -----
        try:
            waveform, _ = load_audio(audio_path, target_sr=16000)
            if waveform is not None:
                mid_feat, final_feat = extract_whisper_encoder_feats(
                    waveform, whisper_model, processor, device, max_duration=args.max_duration
                )
                
                if mid_feat is not None and final_feat is not None:
                    existing_whisper_mid[audio_path] = mid_feat
                    existing_whisper_final[audio_path] = final_feat
                    whisper_success_count += 1
                else:
                    whisper_fail_count += 1
                    
                # 立即释放内存
                del waveform, mid_feat, final_feat
                
            else:
                print(f"音频加载失败: {audio_path}")
                whisper_fail_count += 1
                
        except Exception as e:
            print(f"语音特征提取失败 {audio_path}: {e}")
            whisper_fail_count += 1
            continue

    print(f"\n处理完成统计:")
    print(f"文本嵌入成功: {processed_count}")
    print(f"Whisper特征成功: {whisper_success_count}")
    print(f"Whisper特征失败: {whisper_fail_count}")

    # 最终保存所有数据
    print("正在保存所有数据...")
    try:
        torch.save(existing_text_emb, args.output_text)
        print(f"✅ 已保存完整文本嵌入: {args.output_text} (共 {len(existing_text_emb)} 条)")
        
        whisper_out = {
            "mid": existing_whisper_mid,
            "final": existing_whisper_final
        }
        torch.save(whisper_out, args.output_whisper)
        print(f"✅ 已保存完整Whisper特征: {args.output_whisper} (共 {len(existing_whisper_mid)} 条)")
        
    except Exception as e:
        print(f"❌ 保存最终文件失败: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Input jsonl with audio_path and text")
    parser.add_argument("--model_dir", type=str, required=True, help="CosyVoice1 model dir")
    parser.add_argument("--output_text", type=str, required=True, help="Output .pt for CosyVoice text embeddings")
    parser.add_argument("--output_whisper", type=str, required=True, help="Output .pt for Whisper features")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max audio length (seconds) to process")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Ratio of dataset to use (e.g. 0.1 for 10%)")
    
    main(parser.parse_args())