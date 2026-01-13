# Prefix-SFT 和 GRPO 梯度占比分析

## 总损失函数

```python
loss = prefix_sft_coeff * prefix_sft_loss + grpo_loss
```

其中：
- `prefix_sft_coeff = 0.2`（默认值，可配置）
- `prefix_sft_loss`：Prefix-SFT 损失
- `grpo_loss`：GRPO 损失（系数为 1.0）

## 损失计算方式

### 1. Prefix-SFT Loss

```python
prefix_sft_loss = -prefix_log_probs.sum() / num_prefix_tokens
```

**数学公式**：
$$\mathcal{L}_{\text{prefix-SFT}} = -\frac{1}{N_{\text{prefix}}} \sum_{t \in \text{prefix}} \log \pi_\theta(\text{token}_t | \text{context})$$

**特点**：
- 只对 prefix tokens 计算
- 是平均每个 prefix token 的负对数概率
- 损失值通常在 0-10 之间（取决于模型置信度）

### 2. GRPO Loss

```python
grpo_loss = -(obj.sum() / num_target_tokens)
```

其中 `obj = min(ratio * A, clipped_ratio * A)`

**数学公式**：
$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{N_{\text{cont}}} \sum_{t \in \text{continuation}} \text{ratio}_t \cdot A_t$$

**特点**：
- 只对 continuation tokens 计算
- 是平均每个 continuation token 的 objective 值
- 损失值取决于优势值 A（通常在 -2 到 2 之间）和 ratio（通常在 0.5 到 1.5 之间）

## 梯度占比分析

### 理论梯度占比

总损失的梯度：
$$\nabla_\theta \mathcal{L}_{\text{total}} = \alpha_{\text{prefix}} \cdot \nabla_\theta \mathcal{L}_{\text{prefix-SFT}} + \nabla_\theta \mathcal{L}_{\text{GRPO}}$$

其中 $\alpha_{\text{prefix}} = 0.2$（`prefix_sft_coeff`）

### 实际梯度贡献

梯度贡献不仅取决于系数，还取决于：

1. **损失值的相对大小**
2. **Token 数量的比例**
3. **损失函数的尺度**

#### 场景分析

**假设场景**：
- Prefix tokens: 20 tokens
- Continuation tokens: 100 tokens
- `prefix_sft_coeff = 0.2`

**情况 1：损失值相近**

假设：
- `prefix_sft_loss = 2.0`（平均每个 prefix token 的负对数概率）
- `grpo_loss = 0.5`（平均每个 continuation token 的 objective）

总损失：
```
L_total = 0.2 * 2.0 + 0.5 = 0.4 + 0.5 = 0.9
```

梯度贡献：
- Prefix-SFT: `0.2 * 2.0 = 0.4`，占比 **44.4%**
- GRPO: `0.5`，占比 **55.6%**

**情况 2：GRPO 损失较大**

假设：
- `prefix_sft_loss = 2.0`
- `grpo_loss = 2.0`（优势值较大时）

总损失：
```
L_total = 0.2 * 2.0 + 2.0 = 0.4 + 2.0 = 2.4
```

梯度贡献：
- Prefix-SFT: `0.2 * 2.0 = 0.4`，占比 **16.7%**
- GRPO: `2.0`，占比 **83.3%**

**情况 3：Prefix-SFT 损失较大**

假设：
- `prefix_sft_loss = 5.0`（模型对 prefix 不确定）
- `grpo_loss = 0.5`

总损失：
```
L_total = 0.2 * 5.0 + 0.5 = 1.0 + 0.5 = 1.5
```

梯度贡献：
- Prefix-SFT: `0.2 * 5.0 = 1.0`，占比 **66.7%**
- GRPO: `0.5`，占比 **33.3%**

## Token 数量对梯度的影响

虽然损失是按 token 平均的，但梯度贡献还取决于 token 数量：

### 梯度公式

对于 Prefix-SFT：
$$\nabla_\theta \mathcal{L}_{\text{prefix-SFT}} = -\frac{\alpha_{\text{prefix}}}{N_{\text{prefix}}} \sum_{t \in \text{prefix}} \nabla_\theta \log \pi_\theta(\text{token}_t)$$

对于 GRPO：
$$\nabla_\theta \mathcal{L}_{\text{GRPO}} = -\frac{1}{N_{\text{cont}}} \sum_{t \in \text{continuation}} A_t \cdot \nabla_\theta \log \pi_\theta(\text{token}_t)$$

### 实际梯度贡献（考虑 token 数量）

假设：
- Prefix tokens: $N_p = 20$
- Continuation tokens: $N_c = 100$

**Prefix-SFT 的梯度贡献**：
- 每个 prefix token 的梯度：$\frac{\alpha_{\text{prefix}}}{N_p} \cdot \nabla_\theta \log \pi_\theta(\text{token})$
- 总贡献：$\alpha_{\text{prefix}} \cdot \frac{1}{N_p} \sum_{t \in \text{prefix}} \nabla_\theta \log \pi_\theta(\text{token}_t)$

**GRPO 的梯度贡献**：
- 每个 continuation token 的梯度：$\frac{1}{N_c} \cdot A_t \cdot \nabla_\theta \log \pi_\theta(\text{token})$
- 总贡献：$\frac{1}{N_c} \sum_{t \in \text{continuation}} A_t \cdot \nabla_\theta \log \pi_\theta(\text{token}_t)$

### 综合占比估算

**典型情况**（假设损失值相近）：
- Prefix-SFT 梯度贡献：$0.2 \times \frac{20}{120} = 0.033$（约 3.3%）
- GRPO 梯度贡献：$1.0 \times \frac{100}{120} = 0.833$（约 83.3%）

**但实际占比取决于**：
1. **损失值的相对大小**（更关键）
2. **Token 数量比例**
3. **优势值的分布**

## 实际训练中的占比

### 经验观察

根据代码和配置，实际训练中：

1. **Prefix tokens 通常较少**（10-30 tokens）
2. **Continuation tokens 较多**（50-200 tokens）
3. **`prefix_sft_coeff = 0.2`** 是一个较小的系数

### 典型占比范围

- **Prefix-SFT 梯度占比**：**10% - 30%**
  - 当 prefix 质量较差时（损失大），占比可能达到 30-40%
  - 当 prefix 质量较好时（损失小），占比可能只有 5-15%

- **GRPO 梯度占比**：**70% - 90%**
  - 这是主要的梯度来源
  - 因为 continuation tokens 数量多，且系数为 1.0

### 调整 `prefix_sft_coeff` 的影响

- **`prefix_sft_coeff = 0.1`**：Prefix-SFT 占比约 5-15%
- **`prefix_sft_coeff = 0.2`**（默认）：Prefix-SFT 占比约 10-30%
- **`prefix_sft_coeff = 0.3`**：Prefix-SFT 占比约 15-40%

## 总结

1. **默认配置**（`prefix_sft_coeff = 0.2`）：
   - Prefix-SFT 梯度占比：**约 10-30%**
   - GRPO 梯度占比：**约 70-90%**

2. **主要影响因素**：
   - 损失值的相对大小（最关键）
   - Token 数量比例
   - `prefix_sft_coeff` 系数

3. **设计意图**：
   - Prefix-SFT 作为辅助损失，引导模型学习高质量 prefix
   - GRPO 作为主要损失，通过强化学习优化整体性能
   - 较小的 `prefix_sft_coeff` 确保 GRPO 占主导，同时让 prefix 得到适当的学习信号


