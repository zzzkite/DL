#!/usr/bin/env python3
import json
import os
from pathlib import Path

def create_jsonl(librispeech_dir, output_jsonl):
    """创建音频和文本配对的jsonl文件"""
    pairs = []
    
    # 遍历LibriSpeech目录结构
    for root, dirs, files in os.walk(librispeech_dir):
        for file in files:
            if file.endswith('.trans.txt'):
                trans_file = os.path.join(root, file)
                with open(trans_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # LibriSpeech格式: "音频ID 转录文本"
                        audio_id, text = line.split(' ', 1)
                        audio_file = os.path.join(root, f"{audio_id}.flac")
                        
                        if os.path.exists(audio_file):
                            pairs.append({
                                "audio_path": audio_file,
                                "text": text
                            })
    
    # 写入jsonl文件
    with open(output_jsonl, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"创建了 {len(pairs)} 个音频-文本对到 {output_jsonl}")

if __name__ == "__main__":
    # 配置路径
    librispeech_train_dir = "LibriSpeech/train-clean-100"  # 修改为你的路径
    librispeech_test_dir = "LibriSpeech/test-clean"       # 修改为你的路径
    
    # 创建训练集和测试集的jsonl文件
    create_jsonl(librispeech_train_dir, "train.jsonl")
    create_jsonl(librispeech_test_dir, "test.jsonl")