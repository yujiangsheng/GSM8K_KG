# GSM8K Knowledge Graph Generator

使用 **Qwen2.5-7B-Instruct** 为 OpenAI GSM8K 数据集生成结构化知识图谱。

## 功能特性

- ✅ **自动设备选择**：GPU > MPS > CPU
- ✅ **分批处理**：支持大规模数据集处理
- ✅ **断点续传**：可从中断处继续
- ✅ **进度保存**：按批次存储结果
- ✅ **答案验证**：错误自动重试
- ✅ **结果合并**：自动合并所有批次

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 处理整个数据集（分批进行）
python gsm8k_full_processor.py

# 指定批次大小和起始批次
python gsm8k_full_processor.py --batch-size 50 --start-batch 0

# 只处理前5个批次
python gsm8k_full_processor.py --max-batches 5

# 仅合并已有批次
python gsm8k_full_processor.py --merge-only
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 100 | 每批处理的问题数量 |
| `--start-batch` | 0 | 开始的批次编号 |
| `--max-batches` | 全部 | 最大处理批次数 |
| `--output-dir` | output | 输出目录 |
| `--merge-only` | - | 仅合并已有批次 |

## 输出结构

```
output/
├── batch_0000.json    # 批次 0 (问题 0-99)
├── batch_0001.json    # 批次 1 (问题 100-199)
├── ...
└── GSM8K_KG.json      # 合并后的完整文件
```

## JSON 格式

```json
{
  "metadata": {
    "dataset": "GSM8K",
    "model": "Qwen2.5-7B-Instruct",
    "total": 7473,
    "correct": 6500,
    "accuracy": "87.00%"
  },
  "knowledge_graphs": [
    {
      "question_id": 0,
      "question": "问题文本",
      "ground_truth_answer": "正确答案",
      "cot": "链式思维推理",
      "solution_steps": "解题步骤",
      "final_answer": "模型答案",
      "problem_type": "问题类型",
      "required_knowledge": "所需知识",
      "status": "correct",
      "attempts": 1
    }
  ]
}
```

## 分阶段处理建议

GSM8K 训练集有 **7473** 个问题。建议分阶段处理：

```bash
# 第一阶段：处理批次 0-9 (问题 0-999)
python gsm8k_full_processor.py --start-batch 0 --max-batches 10

# 第二阶段：处理批次 10-19 (问题 1000-1999)
python gsm8k_full_processor.py --start-batch 10 --max-batches 10

# 继续直到完成所有批次...

# 最后合并
python gsm8k_full_processor.py --merge-only
```

## 时间估计

| 设备 | 每问题耗时 | 100个问题 | 全部7473个 |
|------|-----------|-----------|------------|
| GPU | ~30s | ~50分钟 | ~60小时 |
| MPS | ~60s | ~100分钟 | ~120小时 |
| CPU | ~180s | ~300分钟 | ~370小时 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `gsm8k_full_processor.py` | 主程序（分批处理） |
| `output/` | 输出目录 |
| `example_GSM8K_KG.json` | 示例输出 |
| `requirements.txt` | 依赖列表 |
