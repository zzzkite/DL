# 项目 — 说明文档

本仓库包含用于实现和评估 TASTE（Text-Aligned Speech Tokenization and Embedding）简化方案的代码与脚本，已集成 CosyVoice-300M 模型用于 S3 单元重建评估。


**主要文件**
- **`setup_cosyvoice.py`**: 一键下载并校验 CosyVoice-300M（默认保存到 `CosyVoice-300M` 根目录），会自动解压及创建必要目录。
- **`utt2text_and_feature.py`**: 提取文本嵌入与 Whisper 编码器特征的流水线脚本（支持 `--subset_ratio`、内存保护和断点续传/单次输出模式）。
- **`s3.sh`**: 用于从音频中提取 S3 单元的工具脚本（CosyVoice 仓库内也包含相关工具）。
- **`CosyVoice/`**: CosyVoice 源码子模块（用于加载 LLM、flow、hift 以及前端工具），需要去github手动下载。

**快速开始（5 分钟跑通）**
- **创建并激活 Python 环境**（建议使用 conda）:

```bash
conda create -n cosy python=3.10 -y
conda activate cosy
pip install -r CosyVoice/requirements.txt
pip install modelscope huggingface_hub torchaudio transformers tqdm psutil
```

- **下载/校验 CosyVoice-300M**（默认下载到 `CosyVoice-300M`）：

```bash
python setup_cosyvoice.py
# CosyVoice的源码需要去github下载
# 若需深度校验加载模型，可加 --deep-verify
python setup_cosyvoice.py --deep-verify
```

- **准备数据（示例：使用小子集测试）**
  - 请先准备 `json/train.jsonl`，每行形如：{"audio_path": "LibriSpeech/..../xxx.wav", "text": "转录文本"}
  - 使用脚本 `utt2text_and_feature.py` 提取嵌入与特征（可用 `--subset_ratio 0.1` 仅处理 10%）

```bash
python utt2text_and_feature.py \
  --jsonl json/train.jsonl \
  --model_dir CosyVoice-300M \
  --output_text train_text_emb.pt \
  --output_whisper train_whisper_feats.pt \
  --subset_ratio 0.1
```

输出：
- `train_text_emb.pt` — CosyVoice 文本嵌入（完整模式或分块模式，脚本已支持两种流程）
- `train_whisper_feats.pt` — Whisper 编码器中间特征

**脚本行为说明与重要参数**
- **`--subset_ratio`**: 处理数据集比例（0.1 即 10%）；用于调试与节省资源。
- **内存保护**: 脚本会定期检查系统内存使用率（每 20 条），当内存使用率 > 90% 时会优雅停止并保存当前进度，避免整机卡死。
- **断点续传**: 若输出文件已存在，脚本会在启动时尝试加载已有结果并跳过已处理的音频。
- **单次完整输出 / 分块策略**: 默认脚本已经提供两种版本（分块保存/最终合并与一次性保存），请根据机器内存选择：
  - 内存充足（≥32GB）: 可以使用一次性生成完整文件的模式（脚本默认会在内存允许时采用此模式）。
  - 内存受限: 使用分块模式（脚本早期实现），或把 `--subset_ratio` 调低。


**使用 `model_code.py` 训练与推理**

仓库中已经包含一个完整的训练与推理脚本 `model_code.py`，它实现了：
- 文本—语音交叉注意力聚合器（SimpleTextSpeechAggregator）
- 基于 CosyVoice LLM 的 `CosyVoiceS3Model`（聚合器 + 融合 + LLM 前缀掩码与解码头）
- 数据装载：从 `utt2_output` / `s3_output` 的 `.pt` 文件构建训练/验证样本
- 训练、评估循环与示例推理（并保存 `loss_curve.png`）

重要：`model_code.py` 使用文件顶部的常量作为配置（直接修改最顶部的路径与超参数）：
- `TRAIN_UTT2_S3_PATH` / `TRAIN_UTT2_TEXT_EMB_PATH` / `TRAIN_UTT2_WHISPER_PATH`：训练集特征路径
- `TEST_UTT2_S3_PATH` / `TEST_UTT2_TEXT_EMB_PATH` / `TEST_UTT2_WHISPER_PATH`：测试集特征路径
- `COSYVOICE_MODEL_DIR`：CosyVoice-300M 所在目录（默认 `CosyVoice-300M`）
- `BATCH_SIZE`, `LR`, `NUM_EPOCHS` 等超参数也在文件顶部可调

使用前准备（关键检查项）：
- 确认 `CosyVoice-300M` 已经放在仓库根目录或 `COSYVOICE_MODEL_DIR` 指定位置。
- 确认已经用 `utt2text_and_feature.py` 生成了：
  - `utt2_output/train_text_emb.pt`
  - `utt2_output/train_whisper_feats.pt`
  - `s3_output/train_utt2s3.pt`（S3 目标）

快速运行示例：

```bash
# 激活环境（示例）
conda activate cosy

# 确认文件路径或编辑 model_code.py 顶部常量以匹配你的文件位置
python model_code.py
```

运行效果：
- 脚本会加载训练/测试样本、构建 DataLoader、加载 CosyVoice-300M（`CosyVoice` 代码需在 `CosyVoice/` 下）并创建 `CosyVoiceS3Model`。
- 训练完成后会保存 `loss_curve.png` 并在控制台打印示例推理的真实 S3 与预测 S3 的对比。

调整建议：
- 若想微调 LLM 或只训练部分模块，请在 `model_code.py` 中调整 `CosyVoiceS3Model(..., freeze_llm=...)`。
- 若显存受限：降低 `BATCH_SIZE`，或使用较小的 `subset_ratio` 先跑通流水线。训练时脚本会在周期性位置调用 `gc.collect()` 与 `torch.cuda.empty_cache()`。

评估与报告：
- 在 test 数据上运行 `model_code.py`（确保测试集特征文件存在），脚本会在每个 epoch 后打印评估损失与准确率。
- 若需更详尽的度量（BLEU / Top-k / token-level F1），可在 `eval_one_epoch` 中添加自定义统计并保存日志。


**常见问题 & 故障排查**
- OOM（显存）: 使用 `--subset_ratio` 减少数据；在 `utt2text_and_feature.py` 中已删除大模型模块以节省显存；可用 `TORCH_USE_CUDA_DSA` 或 `CUDA_LAUNCH_BLOCKING=1` 做调试。
- 整机内存升高: 若内存逼近满载，脚本会在 >90% 时自动停止并保存进度；建议使用分块保存或扩展物理内存/交换分区。
- CosyVoice 导入失败: 确认 `CosyVoice` 代码在工作目录中且 `setup_cosyvoice.py --deep-verify` 能成功加载模型。
