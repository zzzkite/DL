#!/bin/bash

# 直接尝试提取Something-Something V2数据集

echo "================================================"
echo "直接提取Something-Something V2数据集"
echo "================================================"

# 创建输出目录
mkdir -p extracted_videos
echo "创建输出目录: extracted_videos"

# 记录开始时间
start_time=$(date +%s)

# 方法1: 直接使用cat命令（官方推荐方法）
echo ""
echo "方法1: 使用官方推荐的cat命令..."
cat 20bn-something-something-v2-00 20bn-something-something-v2-01 | tar -xzv -C extracted_videos

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo ""
    echo "✅ 方法1成功!"
    echo "提取耗时: $duration 秒"
    
    # 统计提取的文件
    file_count=$(find extracted_videos -name "*.webm" | wc -l)
    echo "成功提取 $file_count 个视频文件"
    
    # 显示前10个文件作为示例
    echo ""
    echo "前10个提取的文件:"
    find extracted_videos -name "*.webm" | head -10
    
    exit 0
else
    echo "❌ 方法1失败"
fi

# 方法2: 尝试将第二个文件重命名为.gz后提取
echo ""
echo "方法2: 尝试修复第二个文件扩展名..."
cp 20bn-something-something-v2-01 20bn-something-something-v2-01.gz

echo "检查修复后的文件类型:"
file 20bn-something-something-v2-01.gz

cat 20bn-something-something-v2-00 20bn-something-something-v2-01.gz | tar -xzv -C extracted_videos

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo ""
    echo "✅ 方法2成功!"
    echo "提取耗时: $duration 秒"
    
    file_count=$(find extracted_videos -name "*.webm" | wc -l)
    echo "成功提取 $file_count 个视频文件"
    
    # 清理临时文件
    rm 20bn-something-something-v2-01.gz
    
    exit 0
else
    echo "❌ 方法2失败"
    # 清理临时文件
    rm 20bn-something-something-v2-01.gz
fi

# 方法3: 尝试分别解压再合并
echo ""
echo "方法3: 分别解压再合并..."

# 创建临时目录
mkdir -p temp_extract_00 temp_extract_01

echo "解压第一个文件..."
tar -xzf 20bn-something-something-v2-00 -C temp_extract_00/

if [ $? -eq 0 ]; then
    echo "✅ 第一个文件解压成功"
    count00=$(find temp_extract_00 -name "*.webm" | wc -l)
    echo "从第一个文件提取了 $count00 个视频"
else
    echo "❌ 第一个文件解压失败"
fi

echo "尝试解压第二个文件..."
# 尝试多种解压方式
tar -xzf 20bn-something-something-v2-01 -C temp_extract_01/ 2>/dev/null

if [ $? -ne 0 ]; then
    echo "尝试gzip解压..."
    gzip -dc 20bn-something-something-v2-01 | tar -x -C temp_extract_01/ 2>/dev/null
fi

if [ $? -eq 0 ]; then
    echo "✅ 第二个文件解压成功"
    count01=$(find temp_extract_01 -name "*.webm" | wc -l)
    echo "从第二个文件提取了 $count01 个视频"
    
    # 合并文件
    echo "合并提取的文件..."
    cp -r temp_extract_00/* extracted_videos/ 2>/dev/null
    cp -r temp_extract_01/* extracted_videos/ 2>/dev/null
    
    total_count=$(find extracted_videos -name "*.webm" | wc -l)
    echo "总共提取了 $total_count 个视频文件"
    
    # 清理临时目录
    rm -rf temp_extract_00 temp_extract_01
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "提取耗时: $duration 秒"
    
    exit 0
else
    echo "❌ 第二个文件解压失败"
    # 清理临时目录
    rm -rf temp_extract_00 temp_extract_01
fi

# 方法4: 尝试使用dd修复文件头
echo ""
echo "方法4: 尝试修复文件头..."

# 检查第二个文件的头部
echo "第二个文件头部信息:"
hexdump -C 20bn-something-something-v2-01 | head -5

# 如果是gzip文件但头信息损坏，尝试修复
echo "尝试修复为gzip文件..."
cp 20bn-something-something-v2-01 fixed_01.gz

# 尝试提取修复后的文件
cat 20bn-something-something-v2-00 fixed_01.gz | tar -xzv -C extracted_videos/ 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ 方法4成功!"
    file_count=$(find extracted_videos -name "*.webm" | wc -l)
    echo "成功提取 $file_count 个视频文件"
    rm fixed_01.gz
    exit 0
else
    echo "❌ 方法4失败"
    rm fixed_01.gz
fi

# 如果所有方法都失败
echo ""
echo "================================================"
echo "❌ 所有提取方法都失败了"
echo "可能的原因:"
echo "1. 第二个文件下载不完整或损坏"
echo "2. 文件格式不是预期的gzip/tar格式"
echo "3. 需要重新下载第二个文件"
echo ""
echo "建议重新下载第二个文件，确保下载完整"
echo "下载链接: https://developer.qualcomm.com/software/ai-datasets/something-something"
echo "================================================"

exit 1