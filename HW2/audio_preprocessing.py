#!/usr/bin/env python3
import json
import torchaudio
import os
from tqdm import tqdm

def validate_audio_files(jsonl_file):
    """
    验证音频文件的完整性和基本信息
    """
    print(f"验证音频文件: {jsonl_file}")
    
    # 检查文件是否存在
    if not os.path.exists(jsonl_file):
        print(f"❌ 文件不存在: {jsonl_file}")
        return False
    
    # 读取JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    valid_count = 0
    error_files = []
    sample_rates = {}
    durations = []
    
    for line in tqdm(lines, desc="验证音频文件"):
        data = json.loads(line.strip())
        audio_path = data["audio_path"]
        
        try:
            # 尝试加载音频（不实际重采样）
            waveform, sr = torchaudio.load(audio_path)
            
            # 统计信息
            if sr not in sample_rates:
                sample_rates[sr] = 0
            sample_rates[sr] += 1
            
            # 计算时长
            duration = waveform.shape[1] / sr
            durations.append(duration)
            
            valid_count += 1
            
        except Exception as e:
            error_files.append((audio_path, str(e)))
    
    # 输出统计报告
    print(f"\n📊 验证完成:")
    print(f"  总文件数: {len(lines)}")
    print(f"  有效文件: {valid_count}")
    print(f"  错误文件: {len(error_files)}")
    
    print(f"\n📈 采样率统计:")
    for sr, count in sample_rates.items():
        status = "✓" if sr == 16000 else "⚠ (将在特征提取时重采样)"
        print(f"  {sr}Hz: {count} 个文件 {status}")
    
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\n⏱️ 音频时长统计:")
        print(f"  平均时长: {avg_duration:.2f} 秒")
        print(f"  最短时长: {min(durations):.2f} 秒")
        print(f"  最长时长: {max(durations):.2f} 秒")
    
    if error_files:
        print(f"\n❌ 错误文件列表 (前5个):")
        for file, error in error_files[:5]:
            print(f"  {os.path.basename(file)}: {error}")
    
    return valid_count == len(lines)

def main():
    print("开始音频文件验证...")
    
    # 定义JSONL文件路径
    train_jsonl = "json/train.jsonl"
    test_jsonl = "json/test.jsonl"
    
    # 验证训练集
    train_ok = validate_audio_files(train_jsonl)
    print("=" * 50)
    
    # 验证测试集
    test_ok = validate_audio_files(test_jsonl)
    
    print("=" * 50)
    if train_ok and test_ok:
        print("🎉 所有音频文件验证通过！")
    else:
        print("⚠ 发现一些问题，但特征提取脚本会处理重采样")

if __name__ == "__main__":
    main()