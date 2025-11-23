#!/bin/bash
# extract_test_s3.sh - 提取测试集S3 tokens

LIBRISPEECH_DIR="json"
OUT_DIR="s3_output"
ONNX_PATH="CosyVoice-300M/speech_tokenizer_v1.onnx"
COSYVOICE_TOOLS="CosyVoice/tools/extract_speech_token.py"

echo "开始提取测试集S3 tokens..."

CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"

# 检查必要的文件和目录
if [ ! -d "$LIBRISPEECH_DIR" ]; then
    echo "错误: JSONL目录不存在: $LIBRISPEECH_DIR"
    exit 1
fi

if [ ! -f "$LIBRISPEECH_DIR/test.jsonl" ]; then
    echo "错误: 测试集JSONL文件不存在: $LIBRISPEECH_DIR/test.jsonl"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

echo "计算测试集样本总数..."
TOTAL_SAMPLES=$(wc -l < "$LIBRISPEECH_DIR/test.jsonl")

echo "测试集总样本数: $TOTAL_SAMPLES"
echo "开始提取完整测试集数据..."

echo "生成测试集wav.scp文件..."
python3 -c "
import json
import os
import sys

# 从Bash变量获取总样本数
total_samples = $TOTAL_SAMPLES
print(f'开始处理测试集，共{total_samples}个样本', file=sys.stderr)

valid_count = 0
error_count = 0
with open('$LIBRISPEECH_DIR/test.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line.strip())
            audio_path = data['audio_path']
            
            # 转换为绝对路径
            if not os.path.isabs(audio_path):
                audio_path = os.path.join('$CURRENT_DIR', audio_path)
            
            if os.path.exists(audio_path):
                audio_id = os.path.splitext(os.path.basename(audio_path))[0]
                audio_id = audio_id.replace('/', '-').replace(' ', '_')
                
                print(f'{audio_id} {audio_path}')
                valid_count += 1
                
                if valid_count % 100 == 0:
                    print(f'已处理 {valid_count} 个测试文件...', file=sys.stderr)
            else:
                print(f'警告: 测试音频文件不存在: {audio_path}', file=sys.stderr)
                error_count += 1
                
        except Exception as e:
            print(f'错误: 处理测试集第{i+1}行时出错: {e}', file=sys.stderr)
            error_count += 1
            continue

print(f'测试集处理完成: {valid_count} 个有效文件, {error_count} 个错误', file=sys.stderr)
" > "$OUT_DIR/test_wav.scp"

echo "生成的测试集音频文件数量: $(wc -l < "$OUT_DIR/test_wav.scp")"

if [ ! -s "$OUT_DIR/test_wav.scp" ]; then
    echo "错误: test_wav.scp文件为空"
    exit 1
fi

# 显示wav.scp的前几行进行检查
echo "test_wav.scp文件前5行:"
head -n 5 "$OUT_DIR/test_wav.scp"

# 创建测试集专用目录
TEST_DIR="$OUT_DIR/test_processing"
mkdir -p "$TEST_DIR"

# 复制wav.scp到测试目录（使用标准名称wav.scp）
cp "$OUT_DIR/test_wav.scp" "$TEST_DIR/wav.scp"

echo "开始提取测试集S3 tokens..."
cd "$TEST_DIR" && python3 "../../$COSYVOICE_TOOLS" \
  --dir "." \
  --onnx_path "../../$ONNX_PATH"

if [ $? -eq 0 ]; then
    echo "✅ 测试集S3 tokens提取成功！"
    
    if [ -f "utt2speech_token.pt" ]; then
        mv "utt2speech_token.pt" "../test_utt2s3.pt"
        echo "✅ 测试集输出文件: $OUT_DIR/test_utt2s3.pt"
        
        # 显示统计
        echo "提取统计:"
        python3 -c "
import torch
try:
    data = torch.load('../test_utt2s3.pt')
    print(f'成功提取了 {len(data)} 个测试样本的S3 tokens')
    if len(data) > 0:
        sample_keys = list(data.keys())[:3]  # 显示前3个样本
        for i, key in enumerate(sample_keys):
            tokens = data[key]
            print(f'样本{i+1}: {key} -> {len(tokens)} 个tokens')
            if len(tokens) > 5:
                print(f'  前5个tokens: {tokens[:5].tolist()}')
        print(f'... 还有 {len(data) - 3} 个样本')
except Exception as e:
    print(f'读取测试集统计时出错: {e}')
"
        
        # 清理临时目录
        cd .. && rm -rf "$TEST_DIR"
        echo "✅ 已清理临时目录"
    fi
else
    echo "❌ 测试集S3 tokens提取失败"
    echo "临时目录保留在: $TEST_DIR 用于调试"
    exit 1
fi

echo "🎉 测试集S3 tokens提取完成！共处理了 $(wc -l < "$OUT_DIR/test_wav.scp") 个样本"