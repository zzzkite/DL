#!/usr/bin/env bash
LIBRISPEECH_DIR="json"
OUT_DIR="s3_output"
ONNX_PATH="CosyVoice-300M/speech_tokenizer_v1.onnx"
COSYVOICE_TOOLS="CosyVoice/tools/extract_speech_token.py"

echo "开始S3 tokens提取..."

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

echo "生成wav.scp文件..."
# 使用更可靠的方法：从JSONL文件中提取音频路径
python3 -c "
import json
import os

with open('$LIBRISPEECH_DIR/train.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100:  # 限制处理前100个文件进行测试
            break
        data = json.loads(line.strip())
        audio_path = data['audio_path']
        if os.path.exists(audio_path):
            # 使用文件名（不含扩展名）作为ID
            audio_id = os.path.splitext(os.path.basename(audio_path))[0]
            audio_id = audio_id.replace('/', '-')
            print(f'{audio_id} {audio_path}')
        else:
            print(f'警告: 音频文件不存在: {audio_path}', file=sys.stderr)
" > "$OUT_DIR/wav.scp"

echo "生成的音频文件数量: $(wc -l < "$OUT_DIR/wav.scp")"

# 检查wav.scp是否为空
if [ ! -s "$OUT_DIR/wav.scp" ]; then
    echo "错误: wav.scp文件为空，没有找到任何音频文件"
    echo "请检查JSONL文件中的audio_path是否正确"
    exit 1
fi

# 提取S3 tokens
echo "开始提取S3 tokens..."
python3 "$COSYVOICE_TOOLS" \
  --dir "$OUT_DIR" \
  --onnx_path "$ONNX_PATH"

if [ $? -eq 0 ]; then
    echo "✅ S3 tokens提取成功！"
    echo "输出目录: $OUT_DIR"
    
    # 检查输出文件并重命名为期望的名称
    if [ -f "$OUT_DIR/utt2speech_token.pt" ]; then
        mv "$OUT_DIR/utt2speech_token.pt" "$OUT_DIR/utt2s3.pt"
        echo "✅ 输出文件已重命名为: $OUT_DIR/utt2s3.pt"
    fi
else
    echo "❌ S3 tokens提取失败"
    exit 1
fi