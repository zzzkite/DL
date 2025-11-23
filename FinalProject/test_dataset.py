import csv
import os
from datasets import load_dataset

def simple_test():
    """简单测试，不依赖pandas"""
    print("开始简单测试数据收集...")
    
    try:
        # 尝试加载数据集
        dataset = load_dataset("HuggingFaceM4/something_something_v2", split="train", streaming=True)
        print("数据集加载成功!")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return False
    
    # 只测试获取几个样本
    categories = {
        'move_object': ['moving'],
        'drop_object': ['dropping'], 
        'cover_object': ['covering']
    }
    
    selected_samples = {category: [] for category in categories}
    count = 0
    
    print("尝试获取前5个样本...")
    for sample in dataset:
        count += 1
        if count > 5:  # 只取5个样本测试
            break
            
        text_lower = sample['text'].lower()
        print(f"样本 {count}: {sample['text']}")
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                selected_samples[category].append({
                    'video_id': sample['video_id'],
                    'text': sample['text']
                })
                print(f"✅ 匹配到 {category}")
                break
    
    print("\n=== 测试结果 ===")
    for category, samples in selected_samples.items():
        print(f"{category}: {len(samples)} 个样本")
    
    # 保存为CSV（不使用pandas）
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/simple_test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'video_id', 'text'])
        
        for category, samples in selected_samples.items():
            for sample in samples:
                writer.writerow([category, sample['video_id'], sample['text']])
    
    print("结果已保存到 test_results/simple_test.csv")
    return True

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 简单测试成功!")
    else:
        print("\n❌ 简单测试失败")