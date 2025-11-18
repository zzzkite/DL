#!/usr/bin/env python3
import json
import os
from pathlib import Path

def create_librispeech_jsonl(data_dir, output_file, dataset_name="train"):
    """
    创建LibriSpeech数据集的JSONL文件
    """
    pairs = []
    count = 0
    
    print(f"正在处理 {dataset_name} 数据集...")
    print(f"数据目录: {data_dir}")
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.trans.txt'):
                trans_file_path = os.path.join(root, file)
                print(f"找到转录文件: {trans_file_path}")
                
                # 读取转录文件
                with open(trans_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # LibriSpeech格式: "音频ID 转录文本"
                        parts = line.split(' ', 1)
                        if len(parts) != 2:
                            continue
                            
                        audio_id, text = parts
                        audio_file = os.path.join(root, f"{audio_id}.flac")
                        
                        # 检查音频文件是否存在
                        if os.path.exists(audio_file):
                            pairs.append({
                                "audio_path": audio_file,
                                "text": text
                            })
                            count += 1
                            
                            # 每处理100个文件打印一次进度
                            if count % 100 == 0:
                                print(f"已处理 {count} 个音频-文本对...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 成功创建 {output_file}")
    print(f"📊 总共 {len(pairs)} 个音频-文本对")
    
    return pairs

def main():
    # 配置路径 - 根据你的实际目录
    train_dir = "LibriSpeech/train-clean-100"  # 训练集目录
    test_dir = "LibriSpeech/test-clean"        # 测试集目录
    
    # 检查目录是否存在
    if not os.path.exists(train_dir):
        print(f"❌ 训练集目录不存在: {train_dir}")
        print("请确保数据已下载并放置在正确位置")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ 测试集目录不存在: {test_dir}")
        print("请确保数据已下载并放置在正确位置")
        return
    
    # 创建输出目录
    json_dir = "json"
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
        print(f"创建目录: {json_dir}")
    
    # 创建训练集JSONL
    print("=" * 50)
    train_output = os.path.join(json_dir, "train.jsonl")
    train_pairs = create_librispeech_jsonl(train_dir, train_output, "train-clean-100")
    
    # 创建测试集JSONL
    print("=" * 50)
    test_output = os.path.join(json_dir, "test.jsonl")
    test_pairs = create_librispeech_jsonl(test_dir, test_output, "test-clean")
    
    print("=" * 50)
    print("🎉 数据集准备完成！")
    print(f"训练集: {len(train_pairs)} 个样本 -> {train_output}")
    print(f"测试集: {len(test_pairs)} 个样本 -> {test_output}")
    
    # 显示一些样本示例
    print("\n📝 样本示例:")
    with open(train_output, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只显示前3个样本
                break
            data = json.loads(line.strip())
            print(f"  音频: {os.path.basename(data['audio_path'])}")
            print(f"  文本: {data['text'][:50]}...")  # 只显示前50个字符
            print()

if __name__ == "__main__":
    main()