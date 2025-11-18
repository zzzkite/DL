import torchaudio
import matplotlib.pyplot as plt
import json

def plot_mel_spectrogram(audio_path, text):
    waveform, sr = torchaudio.load(audio_path)
    
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
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec_db[0].numpy(), aspect='auto', origin='lower')
    plt.title(f"Mel Spectrogram: {text}")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"mel_spectrogram_{Path(audio_path).stem}.png")
    plt.close()

# 使用示例
with open("train.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:  # 只可视化前3个样本
            break
        data = json.loads(line)
        plot_mel_spectrogram(data["audio_path"], data["text"])