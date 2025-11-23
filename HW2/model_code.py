#!/usr/bin/env python3
import sys
import os
import gc
import psutil

# 添加 CosyVoice 到 Python 路径 - 使用与utt2相同的设置
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

# 现在导入其他模块
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 尝试导入 CosyVoice
try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    print("✅ CosyVoice 导入成功")
except ImportError as e:
    print(f"❌ CosyVoice 导入失败: {e}")
    print("请确保 CosyVoice 目录存在且包含必要的文件")
    sys.exit(1)


# ======================
#  Config - 更新为你的实际路径
# ======================

# 训练集特征文件
TRAIN_UTT2_S3_PATH = "s3_output/train_utt2s3.pt"  
TRAIN_UTT2_TEXT_EMB_PATH = "utt2_output/train_text_emb.pt"  
TRAIN_UTT2_WHISPER_PATH = "utt2_output/train_whisper_feats.pt"    

# 测试集特征文件  
TEST_UTT2_S3_PATH = "s3_output/test_utt2s3.pt"  
TEST_UTT2_TEXT_EMB_PATH = "utt2_output/test_text_emb.pt"  
TEST_UTT2_WHISPER_PATH = "utt2_output/test_whisper_feats.pt"    

COSYVOICE_MODEL_DIR = "CosyVoice-300M"

S3_PAD_ID = -1
S3_VOCAB_SIZE = 4096
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 10
GRAD_CLIP = 1.0
TRAIN_RATIO = 0.95
IGNORE_ID = -100


def print_memory_usage(device, prefix=""):
    """打印内存使用情况"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"{prefix} GPU内存: 已分配 {allocated:.2f}GB, 保留 {reserved:.2f}GB")
    else:
        memory = psutil.virtual_memory()
        print(f"{prefix} CPU内存: 使用率 {memory.percent}% ({memory.used//1024//1024}MB / {memory.total//1024//1024}MB)")


# ======================
#  CosyVoice LLM wrapper
# ======================

def load_cosyvoice_llm(device):
    """加载CosyVoice LLM模型"""
    print(f"正在加载CosyVoice模型从: {COSYVOICE_MODEL_DIR}")
    cosy = CosyVoice(COSYVOICE_MODEL_DIR)
    # 返回整个 wrapper (TransformerLM 或 Qwen2LM)，以便访问 embedding
    llm_wrapper = cosy.model.llm
    print(f"✅ CosyVoice LLM加载成功")
    return llm_wrapper


class SimpleTextSpeechAggregator(nn.Module):
    """
    文本-语音交叉注意力聚合器
    Q = text_emb         : (B, T_text, D_text)
    K = speech_last      : (B, T_speech, D_last)
    V = speech_mid       : (B, T_speech, D_mid)

    Output:
        z   : (B, T_text, hidden_dim) - 对齐后的语音表示
        att : (B, T_text, T_speech)   - 注意力权重
    """
    def __init__(self, text_dim, speech_last_dim, speech_mid_dim, hidden_dim):
        super().__init__()
        # 三个线性投影层，将不同模态的特征映射到相同的隐藏维度
        self.q_proj = nn.Linear(text_dim, hidden_dim)        # 文本查询投影
        self.k_proj = nn.Linear(speech_last_dim, hidden_dim) # 语音键投影（使用深层特征）
        self.v_proj = nn.Linear(speech_mid_dim, hidden_dim)  # 语音值投影（使用中层特征）
        
        print(f"✅ 聚合器初始化完成: text_dim={text_dim}, speech_last_dim={speech_last_dim}, "
              f"speech_mid_dim={speech_mid_dim}, hidden_dim={hidden_dim}")

    def forward(self, text_emb, speech_last, speech_mid, speech_mask=None):
        """
        前向传播：执行文本到语音的交叉注意力
        
        Args:
            text_emb: (B, T_text, D_text) 文本嵌入
            speech_last: (B, T_speech, D_last) 深层语音特征（用于对齐）
            speech_mid: (B, T_speech, D_mid) 中层语音特征（用于内容重建）
            speech_mask: (B, T_speech) 语音掩码，True表示有效位置
            
        Returns:
            z: (B, T_text, hidden_dim) 对齐后的语音表示
            att: (B, T_text, T_speech) 注意力权重
        """
        # 1) 投影输入到相同的隐藏维度
        Q = self.q_proj(text_emb)      # (B, T_text, hidden_dim)
        K = self.k_proj(speech_last)   # (B, T_speech, hidden_dim) 
        V = self.v_proj(speech_mid)    # (B, T_speech, hidden_dim)
        
        # 2) 计算注意力分数: Q * K^T / sqrt(d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, T_text, T_speech)
        
        # 3) 应用语音掩码（如果提供）
        if speech_mask is not None:
            # 将掩码扩展到与分数相同的维度
            mask = speech_mask.unsqueeze(1).expand(-1, Q.size(1), -1)  # (B, T_text, T_speech)
            # 将填充位置的值设为负无穷，这样softmax后会接近0
            scores = scores.masked_fill(~mask, -1e9)
        
        # 4) 应用softmax获取注意力权重
        att = F.softmax(scores, dim=-1)  # (B, T_text, T_speech)
        
        # 5) 计算加权的值（对齐后的语音表示）
        z = torch.matmul(att, V)  # (B, T_text, hidden_dim)
        
        return z, att


class CosyVoiceS3Model(nn.Module):
    """
    CosyVoice LLM + 聚合器的完整模型
    
    Inputs:
        text_emb    : (B, T_text, D_text)
        speech_last : (B, T_speech, D_last)
        speech_mid  : (B, T_speech, D_mid)
        speech_mask : (B, T_speech) bool
        s3_targets  : (B, T_s3) long
        
    Outputs:
        loss        : scalar 
        logits      : (B, T_text, S3_VOCAB_SIZE)
        attn        : (B, T_text, T_speech)
    """
    def __init__(
        self,
        llm_wrapper,
        text_dim,
        speech_last_dim,
        speech_mid_dim,
        hidden_dim,
        s3_vocab_size,
        s3_pad_id=0,
        freeze_llm=True,
    ):
        super().__init__()
        self.llm_wrapper = llm_wrapper
        self.llm = llm_wrapper.llm  # 内部的 Transformer backbone
        
        # 复用预训练的嵌入层
        self.llm_embedding = llm_wrapper.llm_embedding      # [SOS/EOS, TASK]
        self.speech_embedding = llm_wrapper.speech_embedding # S3 tokens
        
        # 获取 LLM 维度
        llm_input_dim = self.speech_embedding.embedding_dim
        try:
            llm_output_dim = self.llm.output_size()
        except AttributeError:
            # 如果没有 output_size 方法，假设输入输出维度相同 (对于大多数 Transformer)
            llm_output_dim = llm_input_dim
            
        print(f"✅ 使用预训练嵌入: speech_emb_dim={llm_input_dim}, llm_out_dim={llm_output_dim}")
        print(f"✅ 预训练语音词表大小: {self.speech_embedding.num_embeddings}")

        self.aggregator = SimpleTextSpeechAggregator(
            text_dim=text_dim,
            speech_last_dim=speech_last_dim,
            speech_mid_dim=speech_mid_dim,
            hidden_dim=hidden_dim,
        )
        self.s3_pad_id = s3_pad_id
        self.s3_vocab_size = s3_vocab_size
        self.s3_vocab_size_with_eos = s3_vocab_size + 1  # 额外的EOS标记
        
        # 投影层
        self.input_proj = nn.Linear(text_dim, llm_input_dim)
        # 复用预训练的输出投影层 (Decoder Head)
        self.proj = llm_wrapper.llm_decoder
        
        # 融合：添加归一化
        self.ln_text = nn.LayerNorm(text_dim)
        self.ln_z = nn.LayerNorm(hidden_dim)
        # self.fuse_alpha = nn.Parameter(torch.tensor(0.0)) # 移除门控参数，使用直接相加

        # 冻结LLM参数
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            for p in self.llm_embedding.parameters():
                p.requires_grad = False
            for p in self.speech_embedding.parameters():
                p.requires_grad = False
            # 确保 llm_decoder 也被冻结 (如果它包含在 llm_wrapper 中)
            for p in self.proj.parameters():
                p.requires_grad = False
            print("✅ LLM及嵌入层参数已冻结")

        print(f"✅ CosyVoiceS3Model初始化完成")

    def forward(
        self,
        text_emb,
        speech_last,
        speech_mid,
        speech_mask=None,
        text_mask=None,
        s3_targets=None,
        s3_lens=None,
    ):
        """
        前向传播
        
        Args:
            text_emb: (B, T_text, D_text) 文本嵌入
            speech_last: (B, T_speech, D_last) 深层语音特征
            speech_mid: (B, T_speech, D_mid) 中层语音特征
            speech_mask: (B, T_speech) 语音掩码
            text_mask: (B, T_text) 文本掩码
            s3_targets: (B, T_s3) S3目标标记
            s3_lens: (B,) S3序列长度
            
        Returns:
            loss: 标量损失
            logits: (B, L, V+1) 预测logits
            attn: (B, T_text, T_speech) 注意力权重
        """
        device = text_emb.device
        B = text_emb.size(0)
        
        # ========== 步骤1: 聚合 + 融合 ==========
        
        # 1) 调用聚合器进行文本-语音对齐
        z, attn = self.aggregator(text_emb, speech_last, speech_mid, speech_mask)
        
        # 2) 融合文本嵌入和对齐后的语音表示
        # 题目要求: e_combined = v + z
        # 我们保留LayerNorm以确保数值稳定性，直接相加
        fused = self.ln_text(text_emb) + self.ln_z(z)
        # fused形状: (B, T_text, text_dim)
        
        # ========== 文本长度和LLM输入构建 ==========

        if text_mask is not None:
            text_lens = text_mask.sum(dim=1).to(dtype=torch.int32, device=device)
        else:
            text_lens = torch.full(
                (B,),
                fused.size(1),
                dtype=torch.int32,
                device=device,
            )

        # 将融合特征投影到LLM输入空间
        fused_llm = self.input_proj(fused)  # (B, T_text, D_llm_in)

        # 准备前缀嵌入
        sos_eos_emb = self.llm_embedding.weight[0].reshape(1, 1, -1).expand(B, 1, -1)
        task_id_emb = self.llm_embedding.weight[1].reshape(1, 1, -1).expand(B, 1, -1)

        # 处理语音目标标记
        speech_ids = s3_targets.clamp(min=0, max=self.s3_vocab_size - 1)  # (B, T_s3)
        speech_embeds = self.speech_embedding(speech_ids)  # (B, T_s3, D_llm_in)

        # 计算S3序列长度
        if s3_lens is None:
             # Fallback if not provided (though it should be)
             s3_lens = (s3_targets != self.s3_pad_id).sum(dim=1).to(dtype=torch.int32, device=device)
        else:
             s3_lens = s3_lens.to(dtype=torch.int32, device=device)

        # 构建LLM输入序列: [SOS] + 融合特征 + [TASK] + 语音嵌入
        lm_input = torch.cat([sos_eos_emb, fused_llm, task_id_emb, speech_embeds], dim=1)  # (B, L, D)
        lm_input_len = (1 + text_lens + 1 + s3_lens).to(dtype=torch.int32, device=device)  # (B,)

        # 通过LLM
        hidden, _ = self.llm(lm_input, lm_input_len)  # (B, L, H)
        logits = self.proj(hidden)                    # (B, L, V+1)

        # ========== 目标构建和损失计算 ==========
        
        # 构建教师强制目标
        L = lm_input.size(1)
        lm_target = torch.full((B, L), IGNORE_ID, dtype=torch.long, device=device)
        
        for i in range(B):
            prefix_len = 2 + text_lens[i]  # [SOS] + fused_len + [TASK]
            slen = s3_lens[i]
            
            if slen > 0:
                # 修复: 目标应该左移一位 (Next Token Prediction)
                # 输入 [TASK] -> 目标 S0
                # 输入 S0     -> 目标 S1
                # ...
                # 输入 S_last -> 目标 EOS
                
                # 1. 填入 S0 到 S_last (作为 [TASK] 到 S_last-1 的目标)
                lm_target[i, prefix_len - 1 : prefix_len + slen - 1] = s3_targets[i, :slen]
                
                # 2. 填入 EOS (作为 S_last 的目标)
                lm_target[i, prefix_len + slen - 1] = self.s3_vocab_size
        
        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits.view(-1, self.s3_vocab_size_with_eos),
            lm_target.view(-1),
            ignore_index=IGNORE_ID
        )
        
        # 计算准确率
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = lm_target != IGNORE_ID
            correct = (preds[mask] == lm_target[mask]).sum()
            total = mask.sum()
            acc = correct.float() / total.float() if total > 0 else torch.tensor(0.0, device=device)
        
        return loss, logits, attn, acc


# ======================
#  Dataset / DataLoader
# ======================

class S3Dataset(Dataset):
    """S3数据集类"""
    def __init__(self, samples):
        self.samples = samples
        print(f"✅ 数据集初始化: {len(samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """批处理函数"""
    B = len(batch)
    
    # 计算各序列长度
    text_lens = [b["text_emb"].size(0) for b in batch]
    speech_lens = [b["speech_mid"].size(0) for b in batch]
    s3_lens = []
    for b in batch:
        tokens = b["s3_tokens"]
        if torch.is_tensor(tokens):
            s3_lens.append(int(tokens.numel()))
        else:
            s3_lens.append(len(tokens))

    # 找到最大长度用于填充
    max_T_text = max(text_lens)
    max_T_speech = max(speech_lens)
    max_T_s3 = max(s3_lens)

    # 获取特征维度
    text_dim = batch[0]["text_emb"].size(-1)
    d_last = batch[0]["speech_last"].size(-1)
    d_mid = batch[0]["speech_mid"].size(-1)

    # 初始化填充张量
    text_emb = torch.zeros(B, max_T_text, text_dim)
    speech_last = torch.zeros(B, max_T_speech, d_last)
    speech_mid = torch.zeros(B, max_T_speech, d_mid)
    speech_mask = torch.zeros(B, max_T_speech, dtype=torch.bool)
    s3_targets = torch.full((B, max_T_s3), S3_PAD_ID, dtype=torch.long)
    text_mask = torch.zeros(B, max_T_text, dtype=torch.bool)

    # 填充数据
    for i, b in enumerate(batch):
        tt = text_lens[i]
        ts = speech_lens[i]
        ts3 = s3_lens[i]

        text_emb[i, :tt] = b["text_emb"]
        speech_last[i, :ts] = b["speech_last"]
        speech_mid[i, :ts] = b["speech_mid"]
        speech_mask[i, :ts] = True
        
        tokens = b["s3_tokens"]
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens, dtype=torch.long)
        else:
            tokens = tokens.to(dtype=torch.long)
        s3_targets[i, :ts3] = tokens[:ts3]
        text_mask[i, :tt] = True

    return {
        "text_emb": text_emb,
        "speech_last": speech_last,
        "speech_mid": speech_mid,
        "speech_mask": speech_mask,
        "s3_targets": s3_targets,
        "s3_lens": torch.tensor(s3_lens, dtype=torch.long),
        "text_mask": text_mask,
    }


def load_samples(utt2s3_path, utt2text_path, utt2whisper_path, dataset_name="训练集"):
    """加载样本数据"""
    print(f"正在加载{dataset_name}样本数据...")
    
    # 加载三个特征文件
    utt2s3 = torch.load(utt2s3_path, map_location="cpu")
    utt2text = torch.load(utt2text_path, map_location="cpu")
    utt2whisper = torch.load(utt2whisper_path, map_location="cpu")

    # 注意: 不同文件中utt id的格式可能不一致，常见情况是
    # - s3 字典使用短形式 id，如 '3830-12531-0003'
    # - text_emb / whisper 使用文件路径形式，如 'LibriSpeech/.../3830-12531-0003.flac'
    # 为保证匹配，我们建立从规范化id->原始key的映射（规范化为basename去掉扩展名）
    def _norm_key(k):
        try:
            if isinstance(k, str) and ('/' in k or '\\' in k):
                return os.path.splitext(os.path.basename(k))[0]
            return k
        except Exception:
            return k

    map_s3 = { _norm_key(k): k for k in utt2s3.keys() }
    map_text = { _norm_key(k): k for k in utt2text.keys() }
    # whisper 存在嵌套 dict，分别映射 mid/final 的 keys
    whisper_mid = utt2whisper.get('mid', {})
    whisper_final = utt2whisper.get('final', {})
    map_whisper_mid = { _norm_key(k): k for k in whisper_mid.keys() }
    map_whisper_final = { _norm_key(k): k for k in whisper_final.keys() }

    # 取交集
    common_keys = sorted(set(map_s3.keys()) & set(map_text.keys()) & set(map_whisper_mid.keys()) & set(map_whisper_final.keys()))

    samples = []
    skipped_count = 0

    for nk in common_keys:
        # 使用映射取得原始字典中的数据
        s3_tokens = utt2s3.get(map_s3[nk])
        text_emb = utt2text.get(map_text[nk])
        speech_mid = whisper_mid.get(map_whisper_mid[nk])
        speech_last = whisper_final.get(map_whisper_final[nk])

        # 跳过无效数据
        if (s3_tokens is None) or (text_emb is None) or (speech_mid is None) or (speech_last is None):
            skipped_count += 1
            continue
        if (getattr(text_emb, "numel", lambda: 0)() == 0) or (getattr(speech_mid, "numel", lambda: 0)() == 0) or (getattr(speech_last, "numel", lambda: 0)() == 0):
            skipped_count += 1
            continue

        samples.append({
            "utt_id": nk,
            "text_emb": text_emb,
            "speech_mid": speech_mid,
            "speech_last": speech_last,
            "s3_tokens": s3_tokens,
        })

    print(f"✅ {dataset_name}加载完成: {len(samples)} 个有效样本, {skipped_count} 个被跳过")
    return samples


# ======================
#  Train / Eval / Predict
# ======================

def train_one_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    num_batches = len(dataloader)
    
    print(f"开始训练epoch，共{num_batches}个batch...")
    
    for batch_idx, batch in enumerate(dataloader):
        # 监控内存
        if batch_idx % 10 == 0:
            print_memory_usage(device, f"训练Batch {batch_idx}/{num_batches}")
        
        # 将数据移动到设备
        batch_on_device = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_on_device[k] = v.to(device)
            else:
                batch_on_device[k] = v
        
        # 前向传播
        loss, logits, attn, acc = model(**batch_on_device)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # 优化器步进
        optimizer.step()
        
        # 统计有效token数量（忽略padding）
        mask = (batch_on_device["s3_targets"] != S3_PAD_ID)
        batch_tokens = mask.sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        total_acc += acc.item() * batch_tokens
        
        # 每10个batch打印一次进度
        if batch_idx % 10 == 0:
            avg_loss = loss.item()
            print(f"  Batch {batch_idx}/{num_batches} | Loss: {avg_loss:.4f} | Acc: {acc.item():.2%} | Tokens: {batch_tokens}")
        
        # 清理以释放内存
        del batch_on_device, loss, logits, attn, acc
        
        # 定期垃圾回收
        if batch_idx % 20 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # 计算平均每个token的损失
    avg_loss_per_token = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_acc = total_acc / total_tokens if total_tokens > 0 else 0.0
    print(f"✅ 训练完成: 平均损失/Token = {avg_loss_per_token:.4f}, 平均准确率 = {avg_acc:.2%}")
    
    return avg_loss_per_token, avg_acc


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    """评估一个epoch"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    num_batches = len(dataloader)
    
    print(f"开始评估，共{num_batches}个batch...")
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到设备
        batch_on_device = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_on_device[k] = v.to(device)
            else:
                batch_on_device[k] = v
        
        # 前向传播
        loss, logits, attn, acc = model(**batch_on_device)
        
        # 统计有效token数量
        mask = (batch_on_device["s3_targets"] != S3_PAD_ID)
        batch_tokens = mask.sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        total_acc += acc.item() * batch_tokens
        
        # 每10个batch打印一次进度
        if batch_idx % 10 == 0:
            print(f"  评估Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f} | Acc: {acc.item():.2%}")
        
        # 清理
        del batch_on_device, loss, logits, attn, acc
    
    # 计算平均损失
    avg_loss_per_token = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_acc = total_acc / total_tokens if total_tokens > 0 else 0.0
    print(f"✅ 评估完成: 平均损失/Token = {avg_loss_per_token:.4f}, 平均准确率 = {avg_acc:.2%}")
    
    return avg_loss_per_token, avg_acc


@torch.no_grad()
def predict_s3(model, text_emb, speech_last, speech_mid, device, max_steps=200):
    """
    自回归解码生成S3 tokens
    
    Args:
        text_emb: (T_text, D_text) 文本嵌入
        speech_last: (T_speech, D_last) 深层语音特征
        speech_mid: (T_speech, D_mid) 中层语音特征
        device: 计算设备
        max_steps: 最大解码步数
        
    Returns:
        pred_s3: (L,) 生成的S3 tokens
    """
    model.eval()
    
    # 1) 添加批次维度并移动到设备
    text_emb = text_emb.unsqueeze(0).to(device)        # (1, T_text, D_text)
    speech_last = speech_last.unsqueeze(0).to(device)  # (1, T_speech, D_last)
    speech_mid = speech_mid.unsqueeze(0).to(device)    # (1, T_speech, D_mid)
    
    # 2) 创建全True的语音掩码（推理时无填充）
    speech_mask = torch.ones(1, speech_last.size(1), dtype=torch.bool, device=device)
    
    # 3) 使用聚合器+融合获取对齐特征
    z, _ = model.aggregator(text_emb, speech_last, speech_mid, speech_mask)
    # w = torch.sigmoid(model.fuse_alpha)
    # fused = w * model.ln_text(text_emb) + (1 - w) * model.ln_z(z)
    fused = model.ln_text(text_emb) + model.ln_z(z)
    fused_llm = model.input_proj(fused)  # (1, T_text, D_llm_in)
    
    # 4) 构建初始序列: [SOS] + 融合特征 + [TASK]
    sos_eos_emb = model.llm_embedding.weight[0].reshape(1, 1, -1)  # (1, 1, D_llm_in)
    task_id_emb = model.llm_embedding.weight[1].reshape(1, 1, -1)  # (1, 1, D_llm_in)
    
    seq = torch.cat([sos_eos_emb, fused_llm, task_id_emb], dim=1)  # (1, 2+T_text, D_llm_in)
    seq_len = torch.tensor([seq.size(1)], dtype=torch.int32, device=device)
    
    # 5) 自回归解码
    generated_ids = []
    T_text = text_emb.size(1)
    
    for step in range(max_steps):
        # 运行LLM
        hidden, _ = model.llm(seq, seq_len)  # (1, current_len, H)
        
        # 获取最后一步的logits
        last_logits = model.proj(hidden[:, -1:])  # (1, 1, V+1)
        
        # 选择概率最高的token
        next_id = last_logits.argmax(dim=-1).squeeze(-1)  # (1,)
        
        # 检查是否生成EOS
        if next_id.item() == model.s3_vocab_size:  # EOS标记
            break
        
        # 限制ID在有效范围内并嵌入
        next_id_clamped = next_id.clamp(min=0, max=model.s3_vocab_size - 1)
        next_embed = model.speech_embedding(next_id_clamped).unsqueeze(1)  # (1, 1, D_llm_in)
        
        # 添加到序列
        seq = torch.cat([seq, next_embed], dim=1)
        seq_len = torch.tensor([seq.size(1)], dtype=torch.int32, device=device)
        generated_ids.append(next_id_clamped.item())
        
        # 如果生成了太多token（超过文本长度的4倍），提前停止
        if len(generated_ids) >= 4 * T_text:
            break
    
    pred_s3 = torch.tensor(generated_ids, dtype=torch.long)
    print(f"✅ 生成完成: {len(pred_s3)} 个S3 tokens")
    
    return pred_s3


# ======================
#  Main
# ======================

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始内存监控
    print_memory_usage(device, "程序开始")
    
    try:
        # 分别加载训练集和测试集样本
        print("正在加载数据集...")
        train_samples = load_samples(
            TRAIN_UTT2_S3_PATH, 
            TRAIN_UTT2_TEXT_EMB_PATH, 
            TRAIN_UTT2_WHISPER_PATH,
            "训练集"
        )
        
        test_samples = load_samples(
            TEST_UTT2_S3_PATH, 
            TEST_UTT2_TEXT_EMB_PATH, 
            TEST_UTT2_WHISPER_PATH,
            "测试集"
        )
        
        if len(train_samples) == 0:
            print("❌ 没有找到训练样本，请检查文件路径")
            return
        
        if len(test_samples) == 0:
            print("⚠ 没有找到测试样本，将从训练集分割验证集")
            # 如果没有测试集，从训练集分割
            random.shuffle(train_samples)
            n_train = int(len(train_samples) * TRAIN_RATIO)
            train_samples, test_samples = train_samples[:n_train], train_samples[n_train:]
        
        print(f"数据集: 训练集 {len(train_samples)} 样本, 测试集 {len(test_samples)} 样本")
        
        # 创建数据集和数据加载器
        train_ds = S3Dataset(train_samples)
        test_ds = S3Dataset(test_samples)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        # 获取特征维度
        example = train_samples[0]
        text_dim = example["text_emb"].size(-1)
        d_last = example["speech_last"].size(-1)
        d_mid = example["speech_mid"].size(-1)
        
        print(f"特征维度: text_dim={text_dim}, speech_last_dim={d_last}, speech_mid_dim={d_mid}")
        
        # 加载LLM和创建模型
        llm_wrapper = load_cosyvoice_llm(device)
        
        model = CosyVoiceS3Model(
            llm_wrapper=llm_wrapper,
            text_dim=text_dim,
            speech_last_dim=d_last,
            speech_mid_dim=d_mid,
            hidden_dim=text_dim,  # 使用文本维度作为隐藏维度
            s3_vocab_size=S3_VOCAB_SIZE,
            s3_pad_id=S3_PAD_ID,
            freeze_llm=True,         
        ).to(device)
        
        # 打印模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数: 总计 {total_params:,}，可训练 {trainable_params:,}")
        
        # 验证 input_proj 是否可训练
        print(f"input_proj requires_grad: {model.input_proj.weight.requires_grad}")
        print(f"aggregator requires_grad: {model.aggregator.q_proj.weight.requires_grad}")
        
        # 创建优化器（只优化可训练参数）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
        
        # 训练循环
        train_losses = []
        test_losses = []
        
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            train_losses.append(train_loss)
            
            # 在测试集上评估
            test_loss, test_acc = eval_one_epoch(model, test_loader, device)
            test_losses.append(test_loss)
            
            print(f"Epoch {epoch:02d} | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2%} | 测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2%}")
            
            # 每3个epoch进行一次完整的内存清理
            if epoch % 3 == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                print_memory_usage(device, f"Epoch {epoch} 后")
        
        # 绘制训练损失曲线
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
            plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('loss_curve.png')
            print("✅ 损失曲线已保存为 loss_curve.png")
        except ImportError:
            print("⚠ 未安装 matplotlib，跳过绘制损失曲线")
        except Exception as e:
            print(f"⚠ 绘制损失曲线失败: {e}")

        # 训练完成，进行推理示例
        print(f"\n{'='*50}")
        print("训练完成，开始推理示例...")
        print(f"{'='*50}")
        
        if len(test_samples) > 0:
            ex = test_samples[0]
            print(f"使用测试样本进行推理: {ex['utt_id']}")
            
            pred_s3 = predict_s3(
                model,
                ex["text_emb"],
                ex["speech_last"],
                ex["speech_mid"],
                device,
            )
            
            # 对比真实和预测的S3 tokens
            true_s3 = ex["s3_tokens"]
            if torch.is_tensor(true_s3):
                true_s3 = true_s3.tolist()
            
            print(f"真实S3 tokens (前10个): {true_s3[:10]}... (共{len(true_s3)}个)")
            print(f"预测S3 tokens (前10个): {pred_s3[:10].tolist()}... (共{len(pred_s3)}个)")
            
            # 计算准确率（如果长度匹配）
            if len(pred_s3) >= min(10, len(true_s3)):
                match_count = sum(1 for i in range(min(10, len(true_s3))) 
                               if i < len(pred_s3) and pred_s3[i].item() == true_s3[i])
                accuracy = match_count / min(10, len(true_s3))
                print(f"前10个token准确率: {accuracy:.2%}")
        
        print("\n🎉 程序执行完成！")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()