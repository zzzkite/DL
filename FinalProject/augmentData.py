import json
import cv2
import os
import shutil
import csv
import numpy as np
from tqdm import tqdm

# ================= 配置 =================
# 你的原始数据文件夹 (里面应该有 metadata.jsonl)
DATA_ROOT = "./processed_data_512"
# 扩增后的输出文件夹 (脚本会自动创建)
OUTPUT_ROOT = "./augmented_data_512"


# =======================================

def flip_image(img_path, save_path):
    # 这个脚本处理的是 .npy 数组 (frames)，但保留兼容性：如果传入的是图片路径则使用 cv2
    if img_path.lower().endswith('.npy'):
        try:
            arr = np.load(img_path)
            # arr can be (T,H,W,C) or (H,W,C)
            if arr.ndim == 4:
                # (T,H,W,C) -> flip width axis
                flipped = np.flip(arr, axis=2).copy()
            elif arr.ndim == 3:
                flipped = np.flip(arr, axis=1).copy()
            else:
                return False
            np.save(save_path, flipped)
            return True
        except Exception:
            return False
    else:
        img = cv2.imread(img_path)
        if img is None:
            return False
        # 水平翻转: 1
        flipped = cv2.flip(img, 1)
        cv2.imwrite(save_path, flipped)
        return True


def flip_text(text):
    # 简单的文本翻转逻辑
    text = text.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right")
    return text


def main():
    # 清理并创建输出目录
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'metadata'), exist_ok=True)

    # 我们将处理 train/val/test 三个 csv，生成对应的增强 csv（只包含增强样本）
    splits = ['train_samples.csv', 'val_samples.csv', 'test_samples.csv']
    for split in splits:
        src_csv = os.path.join(DATA_ROOT, 'metadata', split)
        dst_csv = os.path.join(OUTPUT_ROOT, 'metadata', split)

        if not os.path.exists(src_csv):
            print(f"⚠️ 未找到 {src_csv}，跳过")
            continue

        with open(src_csv, 'r', newline='', encoding='utf-8') as rf:
            reader = csv.DictReader(rf)
            fieldnames = reader.fieldnames
            rows = list(reader)

        new_rows = []
        print(f"🚀 处理 {split} ({len(rows)} 条样本)...")

        for item in tqdm(rows):
            try:
                category = item['category']
                # 原始路径可能已经是相对路径 like 'processed_data_512/frames/...'
                input_path = item['input_frames_path']
                target_path = item['target_frame_path']

                # 计算源文件绝对路径
                src_input = input_path if os.path.isabs(input_path) else os.path.join(os.getcwd(), input_path)
                src_target = target_path if os.path.isabs(target_path) else os.path.join(os.getcwd(), target_path)

                # 输出目录 per category
                out_dir = os.path.join(OUTPUT_ROOT, 'frames', category)
                os.makedirs(out_dir, exist_ok=True)

                in_basename = os.path.basename(input_path)
                tgt_basename = os.path.basename(target_path)

                out_input_name = 'flip_' + in_basename
                out_target_name = 'flip_' + tgt_basename

                out_input_path = os.path.join(out_dir, out_input_name)
                out_target_path = os.path.join(out_dir, out_target_name)

                # 生成翻转 npy 文件
                ok1 = flip_image(src_input, out_input_path)
                ok2 = flip_image(src_target, out_target_path)

                if not (ok1 and ok2):
                    # 如果失败则跳过该样本
                    # 打印一次性错误
                    print(f"⚠️ 翻转失败: {src_input} 或 {src_target}")
                    continue

                # 组装新的 csv 行（路径用相对 OUTPUT_ROOT 的路径以便 loader 使用）
                new_item = item.copy()
                # 使用相对路径（相对于项目根），例如 'augmented_data_512/frames/move_object/flip_123_input.npy'
                rel_input = os.path.join(OUTPUT_ROOT, 'frames', category, out_input_name)
                rel_target = os.path.join(OUTPUT_ROOT, 'frames', category, out_target_name)
                new_item['input_frames_path'] = rel_input
                new_item['target_frame_path'] = rel_target

                # 翻转文本（简单替换 left/right）
                if 'label' in new_item:
                    new_item['label'] = flip_text(new_item.get('label', ''))

                new_rows.append(new_item)

            except Exception as e:
                print(f"⚠️ 处理样本时出错: {e}")
                continue

        # 写出新的 csv
        if len(new_rows) > 0:
            with open(dst_csv, 'w', newline='', encoding='utf-8') as wf:
                writer = csv.DictWriter(wf, fieldnames=fieldnames)
                writer.writeheader()
                for r in new_rows:
                    writer.writerow(r)

            print(f"✅ 已写入增强数据 ({len(new_rows)}) 到 {dst_csv}")
        else:
            print(f"⚠️ 没有生成增强数据 for {split}")


if __name__ == "__main__":
    main()