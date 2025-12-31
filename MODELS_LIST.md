# MIRA 使用的模型列表

## 📋 模型分类

### 1. 目标模型 (Target Models - 被攻击的模型)

这些是用于攻击测试的主要模型：

#### ⭐ 推荐模型 (CPU友好，已充分测试)

| 模型名称 | 本地目录名 | 大小 | 描述 |
|---------|-----------|------|------|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `smollm2-135m` | 135M | 超轻量级，适合基线测试 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `smollm2-1.7b` | 1.7B | 中等大小 SmolLM，CPU 性能好 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `tinyllama-1.1b` | 1.1B | 基于 LLaMA，适合机制分析 |

#### 其他目标模型

| 模型名称 | 本地目录名 | 大小 | 描述 |
|---------|-----------|------|------|
| `gpt2` | `gpt2` | 117M | 经典基线模型 |
| `gpt2-medium` | `gpt2-medium` | 345M | 中等 GPT-2 变体 |
| `distilgpt2` | `distilgpt2` | 82M | 蒸馏版 GPT-2，快速但能力有限 |
| `EleutherAI/pythia-70m` | `EleutherAI--pythia-70m` | 70M | 非常小的模型 |
| `EleutherAI/pythia-160m` | `EleutherAI--pythia-160m` | 160M | 小但有能力 |
| `EleutherAI/pythia-410m` | `EleutherAI--pythia-410m` | 410M | 中等大小 |
| `EleutherAI/pythia-1b` | `EleutherAI--pythia-1b` | 1B | 1B 参数 |
| `Qwen/Qwen2-0.5B` | `Qwen--Qwen2-0.5B` | 0.5B | Qwen 系列小模型 |
| `Qwen/Qwen2.5-3B` | `Qwen--Qwen2.5-3B` | 3B | Qwen 2.5 系列中等模型 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `deepseek-r1` | 1.5B | DeepSeek R1 蒸馏版 |

### 2. 评估模型 (Judge Models - 用于评估攻击成功)

这些模型用于评估攻击是否成功，**不用于攻击测试**：

| 模型名称 | 本地目录名 | 用途 |
|---------|-----------|------|
| `distilbert-base-uncased-finetuned-sst-2-english` | `distilbert-base-uncased-finetuned-sst-2-english` | 攻击成功判断器 |
| `unitary/toxic-bert` | `unitary--toxic-bert` | 毒性/NSFW 判断器 |
| `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers--all-MiniLM-L6-v2` | 语义相似度计算 |
| `BAAI/bge-base-en-v1.5` | (从 HuggingFace 加载) | 嵌入模型 |

### 3. 数据集 (Datasets)

| 数据集名称 | 本地目录名 | 用途 |
|-----------|-----------|------|
| `tatsu-lab/alpaca` | `alpaca/` | Baseline prompts 数据集 |

---

## 📁 本地已下载的模型

根据 `project/models/` 目录，以下模型已下载到本地：

### 目标模型
- ✅ `smollm2-135m` (HuggingFaceTB/SmolLM2-135M-Instruct)
- ✅ `smollm2-1.7b` (HuggingFaceTB/SmolLM2-1.7B-Instruct)
- ✅ `tinyllama-1.1b` (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- ✅ `gpt2-medium` (gpt2-medium)
- ✅ `distilgpt2` (distilgpt2)
- ✅ `EleutherAI--pythia-160m` (EleutherAI/pythia-160m)
- ✅ `Qwen--Qwen2-0.5B` (Qwen/Qwen2-0.5B)
- ✅ `Qwen--Qwen2.5-3B` (Qwen/Qwen2.5-3B)
- ✅ `deepseek-r1` (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

### 评估模型
- ✅ `distilbert-base-uncased-finetuned-sst-2-english`
- ✅ `unitary--toxic-bert`
- ✅ `sentence-transformers--all-MiniLM-L6-v2`

### 数据集
- ✅ `alpaca/` (tatsu-lab/alpaca)

---

## 🔧 当前测试中使用的模型

### 测试程序 (`test_real_attack_prompts.py`)
- **当前使用**: `EleutherAI/pythia-70m`
- **说明**: 小模型，适合快速测试

### 主程序 (`main.py`)
- **默认推荐**: 
  - `gpt2` (0.5 GB)
  - `EleutherAI/pythia-70m` (0.3 GB)
  - `EleutherAI/pythia-160m` (0.6 GB)

---

## 📊 模型统计

### 按大小分类

**超小模型 (< 100M)**
- EleutherAI/pythia-70m (70M)
- distilgpt2 (82M)

**小模型 (100M - 500M)**
- gpt2 (117M)
- HuggingFaceTB/SmolLM2-135M-Instruct (135M)
- EleutherAI/pythia-160m (160M)
- EleutherAI/pythia-410m (410M)

**中等模型 (500M - 2B)**
- Qwen/Qwen2-0.5B (0.5B)
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B)
- EleutherAI/pythia-1b (1B)
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (1.5B)
- HuggingFaceTB/SmolLM2-1.7B-Instruct (1.7B)

**较大模型 (2B - 5B)**
- Qwen/Qwen2.5-3B (3B)

**大模型 (> 2B)**
- gpt2-medium (345M) - 虽然参数不多，但模型较大

### 按用途分类

**攻击测试模型**: 10+ 个
**评估模型**: 4 个
**数据集**: 1 个

---

## 🚀 使用建议

### 快速测试
- `EleutherAI/pythia-70m` - 最小最快
- `distilgpt2` - 快速但能力有限

### 推荐测试
- `HuggingFaceTB/SmolLM2-135M-Instruct` ⭐ - 超轻量级，适合基线
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` ⭐ - 基于 LLaMA，机制分析好
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` ⭐ - 中等大小，性能好

### 深入研究
- `gpt2-medium` - 经典模型
- `Qwen/Qwen2-0.5B` - Qwen 系列
- `deepseek-r1` - DeepSeek R1 蒸馏版

---

## 📝 模型命名规则

### HuggingFace 名称 → 本地目录名

- `/` → `--` (例如: `EleutherAI/pythia-70m` → `EleutherAI--pythia-70m`)
- 保持其他字符不变

### 查看本地模型

```bash
ls project/models/
```

### 查看模型信息

```python
from mira.utils.model_manager import get_model_info, MODEL_REGISTRY

# 查看所有注册的模型
for name, info in MODEL_REGISTRY.items():
    print(f"{name}: {info['size']} - {info['description']}")
```

---

## 🔄 更新记录

- **2024-12-31**: 初始模型列表
- 当前测试模型: `EleutherAI/pythia-70m`
- 本地已下载: 11 个模型 + 1 个数据集

