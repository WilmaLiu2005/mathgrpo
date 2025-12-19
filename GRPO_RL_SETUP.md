# GRPO (Group Relative Policy Optimization) RL 设置文档

## 概述

本项目实现了 GRPO (Group Relative Policy Optimization) 强化学习算法，用于训练语言模型解决数学推理任务（GSM8K）。GRPO 是一种基于策略梯度的离线强化学习方法，通过组内奖励归一化和可选的动态裁剪机制来稳定训练过程。

## 核心算法特性

### 1. Token-Level Policy Gradient

GRPO 在 **token 级别**计算策略梯度，每个生成的 token 都独立贡献到损失函数中：

```
loss = -mean(log_probs * advantages) / num_target_tokens
```

这意味着：
- 长序列对损失的贡献更大（因为 token 数量多）
- 每个 token 的梯度更新是独立的
- 最终损失是所有生成 token 的平均值

### 2. Reward 归一化策略

#### 基础归一化（默认）
- 按 **prefix（问题）** 分组
- 组内奖励归一化：`normalized_reward = (reward - mean) / (std + 1e-4)`
- 确保不同问题的奖励在同一尺度上

#### 长度分组归一化（可选）
当 `use_length_grouping: true` 时：
- 基于生成序列长度的正态分布进行分组
- 分组规则：
  - **short**: `length < mean - σ`
  - **medium**: `mean - σ ≤ length ≤ mean + σ`
  - **long**: `length > mean + σ`
- 每个组内独立归一化，避免长度偏差影响奖励分布

### 3. Dynamic Adaptive Clipping（动态自适应裁剪）

当 `use_dynamic_clipping: true` 时，使用动态裁剪机制来稳定重要性采样：

#### 核心公式

**Importance Ratio:**
```
ratio = exp(log_probs - old_log_probs)
```

**动态边界计算:**
```
L(x) = 0.5 + 0.5 * sqrt(max(1 - 4*ε_low/q(x), 0))
U(x) = 0.5 + 0.5 * sqrt(1 + 4*ε_high/q(x))
```

其中：
- `q(x) = exp(old_log_probs)` 是旧策略的概率
- `ε_low` 和 `ε_high` 是控制参数

**裁剪后的目标函数:**
```
clipped_ratio = clamp(ratio, L(x), U(x))
obj = min(ratio * advantages, clipped_ratio * advantages)
```

#### 特点
- **自适应**: 裁剪范围根据每个 token 的旧概率动态调整
- **低概率 token**: 允许更大的变化范围（探索）
- **高概率 token**: 限制更严格（稳定）

### 4. KL 散度正则化（可选）

当 `use_kl_penalty: true` 时：
- 需要参考模型（reference model）作为基线
- 计算当前策略与参考策略的 KL 散度
- 添加到损失函数中：`loss += kl_coeff * KL_loss`

**注意**: 启用 KL 正则化会：
- 需要额外的显存（存储 reference model）
- 增加计算开销（需要前向传播 reference model）

## 训练流程

### 1. Rollout（采样阶段）

对每个问题采样 `num_answer_per_question` 个答案：
- 使用当前策略模型进行自回归生成
- 记录每个生成 token 的 log 概率（用于重要性采样）
- 计算奖励（格式奖励 + 答案正确性奖励）

### 2. Reward 归一化

根据配置进行组内奖励归一化：
- 基础模式：按 prefix 分组
- 长度分组模式：按 prefix + 长度桶分组

### 3. Policy Update（策略更新）

#### 批处理策略
- 按序列长度排序（提高批处理效率）
- 使用 micro-batching 处理大批次
- 只对生成的 token 计算梯度（prefix 部分 mask 掉）

#### 损失计算
```python
# 基础策略梯度
obj = log_probs * advantages

# 如果启用 dynamic clipping
if use_dynamic_clipping:
    ratio = exp(log_probs - old_log_probs)
    clipped_ratio = clamp(ratio, lower_bound, upper_bound)
    obj = min(ratio * advantages, clipped_ratio * advantages)

# Token-level 平均
loss = -mean(obj * target_masks) / num_target_tokens

# 如果启用 KL 正则化
if use_kl_penalty:
    loss += kl_coeff * KL_loss
```

#### 优化器更新
- 梯度裁剪：`max_grad_norm = 1.0`
- AdamW 优化器
- 学习率：`1e-5`

## 配置参数详解

### 模型配置
```yaml
model:
  pretrained_model_path: "..."  # 预训练模型路径
  device: "cuda"                 # 设备
  dtype: "bfloat16"              # 数据类型
```

### 训练配置

#### 基础参数
```yaml
training:
  batch_size: 32                 # 总批次大小
  num_questions_per_batch: 16    # 每个批次的问题数
  # 每个问题采样答案数 = batch_size / num_questions_per_batch = 2
  
  micro_batch_size: 2            # 微批次大小（用于梯度累积）
  max_grad_norm: 1.0             # 梯度裁剪阈值
  learning_rate: 1.0e-5          # 学习率
```

#### 优化选项

**Dynamic Clipping:**
```yaml
use_dynamic_clipping: true   # 是否启用动态裁剪
epsilon_low: 0.3              # 下界控制参数（值越大，下界越严格）
epsilon_high: 0.3             # 上界控制参数（值越大，上界越宽松）
```

**KL 正则化:**
```yaml
use_kl_penalty: false         # 是否启用 KL 散度正则化
kl_coeff: 0.05                # KL 散度系数（仅在启用时生效）
```

**Reward 归一化:**
```yaml
use_length_grouping: false   # 是否使用长度分组归一化
```

### 数据配置
```yaml
data:
  path: "/path/to/gsm8k"      # 数据路径
  config_name: "main"          # 数据集配置（"main" 或 "socratic"）
  test_size: 128               # 测试集大小
```

## 当前配置状态

根据 `config.yaml`，当前设置：

✅ **启用**:
- Dynamic Clipping (`use_dynamic_clipping: true`)
- 基础 Reward 归一化（按 prefix 分组）

❌ **禁用**:
- KL 散度正则化 (`use_kl_penalty: false`)
- 长度分组归一化 (`use_length_grouping: false`)

## 奖励函数

### 组成
```
total_reward = 0.1 * format_reward + answer_reward
```

### Format Reward
- 检查输出格式：`<think>...</think><answer>...</answer>`
- 完整格式：1.0
- 部分格式：0.1 (有 reasoning) + 0.5 (有 answer)

### Answer Reward
- 提取 `<answer>` 标签中的数值
- 与标准答案（`#### 数值`）比较
- 数值相等：1.0
- 否则：0.0

## 监控指标

训练过程中记录：
- `loss`: 策略梯度损失
- `kl_loss`: KL 散度损失（如果启用）
- `grad_norm`: 梯度范数
- `entropy`: 策略熵
- `mean_reward`: 平均奖励
- `success_rate`: 答案正确率
- `mean_response_len`: 平均响应长度

## 使用建议

### 1. 只使用 Dynamic Clipping（当前配置）
```yaml
use_dynamic_clipping: true
use_kl_penalty: false
use_length_grouping: false
```
- ✅ 节省显存（不需要 reference model）
- ✅ 训练稳定
- ✅ 适合大多数场景

### 2. 启用 KL 正则化
```yaml
use_dynamic_clipping: true
use_kl_penalty: true
use_length_grouping: false
```
- ⚠️ 需要额外显存
- ✅ 更好的策略约束
- ✅ 适合需要严格控制策略偏离的场景

### 3. 启用长度分组
```yaml
use_dynamic_clipping: true
use_kl_penalty: false
use_length_grouping: true
```
- ✅ 避免长度偏差
- ✅ 适合生成长度差异较大的任务

### 4. 组合使用
```yaml
use_dynamic_clipping: true
use_kl_penalty: true
use_length_grouping: true
```
- ⚠️ 需要最多显存
- ✅ 最稳定的训练配置

## 技术细节

### Token-Level vs Sequence-Level

当前实现是 **token-level** 的：
- 每个 token 独立计算损失
- 最终损失 = 所有 token 损失的平均值
- 长序列贡献更大

如果要改为 sequence-level：
```python
# Sequence-level 实现（当前未使用）
obj_per_seq = (obj * target_masks).sum(dim=1) / target_masks.sum(dim=1)
obj = obj_per_seq.mean()
```

### 显存优化

- 使用 `torch.autocast` 进行混合精度训练
- Micro-batching 减少峰值显存
- 按长度排序提高批处理效率
- 可选：不创建 reference model（当 `use_kl_penalty=false`）

## 文件结构

```
GRPO-zero/
├── grpo.py              # 核心算法实现
│   ├── rollout()        # 采样阶段
│   ├── normalize_rewards_per_group()  # 奖励归一化
│   └── update_policy()  # 策略更新
├── train.py             # 训练主循环
├── config.yaml          # 配置文件
├── countdown_task.py    # 任务和奖励函数
└── data_types.py        # 数据类型定义
```

## 参考文献

- GRPO: Group Relative Policy Optimization
- Dynamic Adaptive Clipping for Stable Policy Optimization
- Token-Level vs Sequence-Level Policy Gradient Methods

