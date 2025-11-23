#!/usr/bin/env bash
LIBRISPEECH_DIR="json"
OUT_DIR="s3_output"
ONNX_PATH="CosyVoice-300M/speech_tokenizer_v1.onnx"
COSYVOICE_TOOLS="CosyVoice/tools/extract_speech_token.py"

echo "开始S3 tokens提取..."

# 获取当前目录的绝对路径
CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"

# 检查必要的文件和目录
if [ ! -d "$LIBRISPEECH_DIR" ]; then
    echo "错误: JSONL目录不存在: $LIBRISPEECH_DIR"
    exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
    echo "错误: ONNX文件不存在: $ONNX_PATH"
    exit 1
fi

if [ ! -f "$COSYVOICE_TOOLS" ]; then
    echo "错误: CosyVoice工具不存在: $COSYVOICE_TOOLS"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

echo "计算训练集样本总数..."
TOTAL_SAMPLES=$(wc -l < "$LIBRISPEECH_DIR/train.jsonl")
SUBSET_SIZE=$((TOTAL_SAMPLES / 10))

echo "训练集总样本数: $TOTAL_SAMPLES"
echo "提取1/10样本数: $SUBSET_SIZE"

echo "生成wav.scp文件..."
# 使用绝对路径生成wav.scp
python3 -c "
import json
import os
import sys

# 计算要提取的样本数量
total_samples = $TOTAL_SAMPLES
subset_size = $SUBSET_SIZE

print(f'总样本数: {total_samples}, 提取1/10: {subset_size}个样本', file=sys.stderr)

valid_count = 0
with open('$LIBRISPEECH_DIR/train.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if valid_count >= subset_size:
            break
        try:
            data = json.loads(line.strip())
            audio_path = data['audio_path']
            
            # 转换为绝对路径
            if not os.path.isabs(audio_path):
                audio_path = os.path.join('$CURRENT_DIR', audio_path)
            
            # 确保音频文件存在
            if os.path.exists(audio_path):
                # 使用安全的ID生成方法
                audio_id = os.path.splitext(os.path.basename(audio_path))[0]
                audio_id = audio_id.replace('/', '-').replace(' ', '_')
                
                # 使用绝对路径
                print(f'{audio_id} {audio_path}')
                valid_count += 1
                
                # 每处理100个文件打印一次进度
                if valid_count % 100 == 0:
                    print(f'已处理 {valid_count}/{subset_size} 个文件...', file=sys.stderr)
            else:
                print(f'警告: 音频文件不存在: {audio_path}', file=sys.stderr)
                
        except Exception as e:
            print(f'错误: 处理第{i+1}行时出错: {e}', file=sys.stderr)
            continue

print(f'最终处理了 {valid_count} 个有效文件', file=sys.stderr)
" > "$OUT_DIR/wav.scp"

echo "生成的音频文件数量: $(wc -l < "$OUT_DIR/wav.scp")"

# 检查wav.scp是否为空
if [ ! -s "$OUT_DIR/wav.scp" ]; then
    echo "错误: wav.scp文件为空，没有找到任何音频文件"
    echo "请检查JSONL文件中的audio_path是否正确"
    exit 1
fi

# 显示wav.scp的前几行进行检查
echo "wav.scp文件前5行:"
head -n 5 "$OUT_DIR/wav.scp"

# 直接在当前目录处理，不使用临时目录
echo "开始提取S3 tokens..."
cd "$OUT_DIR" && python3 "../$COSYVOICE_TOOLS" \
  --dir "." \
  --onnx_path "../$ONNX_PATH"

if [ $? -eq 0 ]; then
    echo "✅ S3 tokens提取成功！"
    
    # 检查输出文件并重命名为期望的名称
    if [ -f "utt2speech_token.pt" ]; then
        mv "utt2speech_token.pt" "train_utt2s3.pt"
        echo "✅ 输出文件已重命名为: $OUT_DIR/train_utt2s3.pt"
        
        # 显示提取统计信息
        echo "提取统计:"
        python3 -c "
import torch
data = torch.load('train_utt2s3.pt')
print(f'成功提取了 {len(data)} 个样本的S3 tokens')
if len(data) > 0:
    sample_key = list(data.keys())[0]
    sample_tokens = data[sample_key]
    print(f'示例: {sample_key} -> {len(sample_tokens)} 个tokens')
    if len(sample_tokens) > 10:
        print(f'前10个tokens: {sample_tokens[:10].tolist()}')
    else:
        print(f'所有tokens: {sample_tokens.tolist()}')
"
    fi
else
    echo "❌ S3 tokens提取失败"
    exit 1
fi

echo "🎉 S3 tokens提取完成！使用了训练集的1/10数据"