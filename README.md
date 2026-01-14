# Exploring GRPO Optimization Beyond DAPO: A Systematic Study on Known Limitations

本项目是对 Group Relative Policy Optimization (GRPO) 算法的系统性研究，旨在探索 GRPO 优化方法并系统性地研究已知限制。

## 项目概述

本项目实现了 GRPO 算法，并在 GSM8K 数学问题求解任务上进行了实验。项目包含完整的训练流程、评估系统以及多种优化技术的实现，包括动态裁剪、难度感知优势缩放、相似度加权奖励等。

## 代码结构

```
GRPO-zero/
├── train.py                 # 主训练脚本
├── grpo.py                  # GRPO 算法核心实现（rollout 和 update_policy）
├── qwen2_model.py          # Qwen2 Transformer 模型实现
├── optimizer.py            # 内存高效的 AdamW 优化器
├── countdown_task.py       # GSM8K 数据集和任务相关功能
├── generate_prefixes.py    # Prefix 生成脚本（用于 prefix 训练模式）
├── openrouter_client.py    # OpenRouter API 客户端（用于调用外部模型）
├── tokenizer.py            # 分词器实现
├── data_types.py           # 数据类型定义（Episode, MiniBatch）
├── config.yaml             # 配置文件
└── requirements.txt        # 依赖包列表
```

## 核心功能模块

### 1. GRPO 算法实现 (`grpo.py`)

- **`rollout()`**: 执行策略采样，为每个问题生成多个答案
  - 支持 prefix 训练模式（使用预生成的思考前缀）
  - 支持温度采样和多种采样策略
  - 返回完整的 Episode 信息（token IDs、log probabilities、rewards 等）

- **`update_policy()`**: 执行策略更新
  - 动态裁剪（Dynamic Clipping）机制
  - 难度感知优势缩放（Difficulty-aware Advantage Scaling）
  - 相似度加权奖励（Similarity-weighted Reward）
  - KL 散度正则化
  - 长度分组奖励归一化

### 2. 训练脚本 (`train.py`)

- 完整的训练循环实现
- 支持多轮训练（epochs）和步数限制（max_steps）
- 定期评估和检查点保存
- 集成 WandB 日志记录
- 支持长度分组采样策略

### 3. 模型实现 (`qwen2_model.py`)

- Qwen2 Transformer 架构实现
- 支持 bfloat16 和 float16 精度
- 高效的注意力机制和前向传播

### 4. 任务模块 (`countdown_task.py`)

- **`GSM8KDataset`**: GSM8K 数据集加载器
  - 支持训练集和测试集
  - 支持 prefix 数据加载
  - 自动批处理和数据整理

- **`reward_function()`**: 奖励函数
  - 基于答案正确性的奖励计算
  - 支持多种奖励策略

- **Prefix 生成功能**:
  - `generate_prefix_with_deepseek()`: 使用 DeepSeek API 生成 prefix
  - `generate_prefix_with_3b()`: 使用本地 3B 模型生成 prefix
  - `clean_prefix_text()`: 清理和规范化 prefix 文本

### 5. Prefix 生成工具 (`generate_prefixes.py`)

- 批量生成 prefix 数据
- 支持并发生成（DeepSeek API 和本地 3B 模型）
- 自动截断和清理
- 输出 JSON 格式的 prefix 文件供训练使用

### 6. 优化器 (`optimizer.py`)

- **`MemoryEfficientAdamW`**: 内存高效的 AdamW 实现
  - 支持将优化器状态存储在 CPU 上以节省 GPU 内存
  - 保持参数和梯度在 GPU 上

### 7. 数据类型 (`data_types.py`)

- **`Episode`**: 存储单个采样轨迹的所有信息
  - 前缀、生成文本、token IDs
  - 奖励信息和 log probabilities
  - Prefix 训练模式相关字段

- **`MiniBatch`**: 批处理数据结构
  - 问题和答案列表
  - Prefix 数据（如果启用）

## 主要特性

### GRPO 优化技术

1. **动态裁剪 (Dynamic Clipping)**
   - 根据样本难度动态调整裁剪范围
   - 可配置的 epsilon 上下界

2. **难度感知优势缩放 (Difficulty-aware Advantage Scaling)**
   - 跨组优势归一化
   - 考虑不同难度组之间的差异

3. **相似度加权奖励 (Similarity-weighted Reward)**
   - 基于隐空间相似度的奖励重加权
   - 可配置的混合系数和温度参数

4. **KL 散度正则化**
   - 防止策略偏离初始策略过远
   - 可配置的正则化系数

5. **长度分组奖励归一化**
   - 按生成长度分组进行奖励归一化
   - 减少长度偏差对训练的影响

### Prefix 训练模式

- 使用预生成的思考前缀引导模型生成
- 支持 DeepSeek（teacher）和 3B（student）两种 prefix 来源
- Prefix dropout 机制，提高泛化能力
- Prefix-SFT loss，对齐 prefix 分布

## 使用方法

### 1. 配置环境

安装依赖：
```bash
pip install -r requirements.txt
```

### 2. 配置文件

编辑 `config.yaml`，设置：
- 模型路径 (`model.pretrained_model_path`)
- 数据路径 (`data.path`)
- 训练超参数
- GRPO 算法参数

### 3. 生成 Prefix（可选）

如果使用 prefix 训练模式：
```bash
python generate_prefixes.py --config config.yaml --output prefixes.json
```

然后在 `config.yaml` 中设置：
```yaml
training:
  enable_prefix: true
  prefix_file: "prefixes.json"
```

### 4. 开始训练

```bash
python train.py --config config.yaml
```

## 配置说明

主要配置项包括：

- **模型配置**: 模型路径、设备、数据类型
- **数据配置**: 数据集路径、测试集大小
- **训练配置**: 批次大小、学习率、训练轮数
- **GRPO 参数**: 动态裁剪、优势缩放、相似度加权等开关和参数
- **Prefix 配置**: Prefix 训练模式相关参数
- **WandB 配置**: 实验跟踪和日志记录

详细配置说明请参考 `config.yaml` 文件中的注释。

## 依赖

主要依赖包：
- PyTorch
- NumPy
- PyYAML
- WandB
- Pandas
- tqdm
- HuggingFace Transformers (用于模型加载)

完整依赖列表请参见 `requirements.txt`。

## 许可证

详见 `LICENSE` 文件。

