import torchaudio
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def plot_mel_spectrogram(audio_path, text, output_dir="mel_spectrograms"):
    """
    绘制并保存梅尔频谱图
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 如果是立体声，转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 创建梅尔频谱图
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        mel_spec = mel_transform(waveform)
        
        # 转换为对数尺度
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # 绘图
        plt.figure(figsize=(12, 4))
        plt.imshow(mel_spec_db[0].detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Mel Spectrogram: {text[:60]}...", fontsize=10)  # 限制标题长度
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        
        # 保存图片
        filename = f"mel_{Path(audio_path).stem}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 处理失败 {audio_path}: {e}")
        return False

def main():
    print("开始生成梅尔频谱图...")
    
    # 定义JSONL文件路径
    jsonl_file = "json/train.jsonl"  # 根据你的实际路径修改
    
    # 检查文件是否存在
    if not os.path.exists(jsonl_file):
        print(f"❌ JSONL文件不存在: {jsonl_file}")
        return
    
    # 从训练集中选择几个样本进行可视化
    sample_count = 5
    visualized = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_count:
                break
                
            data = json.loads(line.strip())
            audio_path = data["audio_path"]
            text = data["text"]
            
            print(f"处理 [{i+1}/{sample_count}]: {os.path.basename(audio_path)}")
            if plot_mel_spectrogram(audio_path, text):
                visualized += 1
    
    print(f"\n🎉 成功生成 {visualized}/{sample_count} 个梅尔频谱图")
    print(f"图片保存在: mel_spectrograms/ 目录")

if __name__ == "__main__":
    main()