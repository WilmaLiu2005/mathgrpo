# GRPO 算法公式文档

本文档详细说明了本代码框架中实现的 GRPO (Group Relative Policy Optimization) 算法的数学公式。

## 目录

1. [GRPO 算法流程](#grpo-算法流程)
2. [核心概念](#核心概念)
3. [GRPO 目标函数](#grpo-目标函数)
4. [动态裁剪 (Dynamic Clipping)](#动态裁剪-dynamic-clipping)
5. [固定裁剪 (PPO-style Clipping)](#固定裁剪-ppo-style-clipping)
6. [KL 散度惩罚](#kl-散度惩罚)
7. [Prefix-SFT 损失](#prefix-sft-损失)
8. [奖励归一化](#奖励归一化)
9. [总损失函数](#总损失函数)

---

## GRPO 算法流程

GRPO 算法的核心思想是：对于每个问题，随机采样多个答案，然后将答案的优势定义为归一化的奖励。这样就不需要价值估计网络了。具体算法流程如下：

1. **采样问题**: 对于每个训练步骤，随机采样 $N$ 个问题 $q_1, q_2, \cdots, q_N$。

2. **采样答案**: 对于每个问题 $q_i$，采样 $M$ 个答案 $a_{i,1}, a_{i,2}, \cdots, a_{i,M}$。

3. **计算奖励**: 计算每个答案 $a_{i,j}$ 的奖励 $r_{i,j}$。

4. **计算统计量**: 对于每个问题 $q_i$，计算其 $M$ 个答案的奖励均值和标准差：

$$
\begin{aligned}
\mu_i &\leftarrow \text{mean}(r_{i,1}, r_{i,2}, \cdots, r_{i,M}) \\
\sigma_i &\leftarrow \text{std}(r_{i,1}, r_{i,2}, \cdots, r_{i,M})
\end{aligned}
$$

5. **计算优势**: 对于答案 $a_{i,j}$ 中的每个 token $t$，计算优势为：

$$A_{i,j}[t] \leftarrow \frac{r_{i,j} - \mu_i}{\sigma_i}$$

注意：对于同一个答案中的所有 token，优势值都是相同的（都等于归一化的奖励）。

6. **计算策略梯度**: 使用 PPO 代理目标计算策略梯度。为了简化，我们每次迭代只进行一次策略更新，此时 PPO 目标的梯度等价于以下策略梯度估计（每个 token）：

$$\nabla_\theta \log \pi_\theta(a_{i,j}[t]) \cdot A_{i,j}[t]$$

7. **更新策略**: 使用梯度更新策略网络 $\pi_\theta$，然后回到步骤 1。

---

## 核心概念

### 符号说明

- $\pi_\theta$: 当前策略（待优化的模型）
- $\pi_{\text{old}}$: 旧策略（rollout 时使用的策略）
- $q_i$: 第 $i$ 个问题
- $a_{i,j}$: 问题 $q_i$ 的第 $j$ 个答案
- $a_{i,j}[t]$: 答案 $a_{i,j}$ 中的第 $t$ 个 token
- $r_{i,j}$: 答案 $a_{i,j}$ 的奖励
- $\mu_i$: 问题 $q_i$ 的所有答案的奖励均值
- $\sigma_i$: 问题 $q_i$ 的所有答案的奖励标准差
- $A_{i,j}[t]$: 答案 $a_{i,j}$ 中第 $t$ 个 token 的优势
- $x$: 输入序列（prompt + prefix + continuation）
- $y$: 生成的 continuation tokens

### 重要性采样比率

重要性采样比率定义为：

$$\text{ratio}(x) = \frac{\pi_\theta(x)}{\pi_{\text{old}}(x)} = \exp(\log \pi_\theta(x) - \log \pi_{\text{old}}(x))$$

在代码中：
```python
ratio = torch.exp(log_probs - old_log_probs)
```

其中：
- `log_probs`: 当前策略的对数概率 $\log \pi_\theta(x)$
- `old_log_probs`: 旧策略的对数概率 $\log \pi_{\text{old}}(x)$

---

## GRPO 目标函数

GRPO 的核心目标函数基于策略梯度。对于每个答案 $a_{i,j}$ 中的每个 token $t$，策略梯度为：

$$\nabla_\theta \log \pi_\theta(a_{i,j}[t]) \cdot A_{i,j}[t]$$

其中 $A_{i,j}[t] = \frac{r_{i,j} - \mu_i}{\sigma_i}$ 是归一化的优势。

### 标准 GRPO 公式（双重平均）

标准 GRPO 公式使用双重平均结构：

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \left\{ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \cdot A_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \cdot A_{i,t} \right) \right] - \beta \cdot D_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}] \right\}$$

其中：
- 先对每个答案内的 token 求平均：$\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$
- 再对所有答案求平均：$\frac{1}{G} \sum_{i=1}^G$

### 本实现使用的全局平均形式

本代码实现使用**全局平均**形式（所有 token 等权重），对应以下公式：

$$J_{\text{GRPO}}^{\text{global}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \left\{ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \left[ \min\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \cdot A_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \cdot A_{i,t} \right) \right] - \beta \cdot \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} D_{\text{KL}}[\pi_\theta(\cdot | q, o_{i,<t}) || \pi_{\text{ref}}(\cdot | q, o_{i,<t})] \right\}$$

**主要区别**：
- **标准公式**：$\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$ （双重平均，每个答案权重相等）
- **全局平均**：$\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|}$ （全局平均，每个 token 权重相等）

**当所有答案长度相等时，两者等价**；当答案长度不等时，全局平均会给长答案更大的权重。

### 代码实现

在代码实现中，我们只对 continuation tokens 计算损失（不包括 prompt 和 prefix）：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{N_{\text{tokens}}} \sum_{i \in \text{continuation}} \text{ratio}_i \cdot A_i$$

其中：
- $N_{\text{tokens}}$ 是所有 continuation tokens 的总数
- $\text{ratio}_i = \frac{\pi_\theta(x_i)}{\pi_{\text{old}}(x_i)}$ 是重要性采样比率
- $A_i$ 是归一化的优势（对于同一答案中的所有 token 都相同）

---

## 动态裁剪 (Dynamic Clipping)

当 `use_dynamic_clipping=True` 时，使用动态自适应裁剪来限制策略更新的幅度。

### 动态边界计算

对于每个 token 位置，计算动态下界和上界：

**下界 (Lower Bound):**

$$L(x) = 0.5 + 0.5 \cdot \sqrt{\max\left(1 - \frac{4 \epsilon_{\text{low}}}{q(x)}, 0\right)}$$

**上界 (Upper Bound):**

$$U(x) = 0.5 + 0.5 \cdot \sqrt{1 + \frac{4 \epsilon_{\text{high}}}{q(x)}}$$

其中：
- $\epsilon_{\text{low}}$: 下界参数 (`epsilon_low`)
- $\epsilon_{\text{high}}$: 上界参数 (`epsilon_high`)
- $q(x) = \exp(\log \pi_{\text{old}}(x))$: 旧策略的概率

### 裁剪后的目标函数

$$\mathcal{L}_{\text{GRPO}}^{\text{dynamic}} = -\frac{1}{N_{\text{tokens}}} \sum_{i \in \text{continuation}} \min\left(\text{ratio}_i \cdot A_i, \text{clipped\_ratio}_i \cdot A_i\right)$$

其中：

$$\text{clipped\_ratio}_i = \text{clip}(\text{ratio}_i, L(x_i), U(x_i))$$

代码实现：
```python
lower_bound = 0.5 + 0.5 * torch.sqrt(torch.clamp(1 - 4 * epsilon_low / q_x, min=0.0))
upper_bound = 0.5 + 0.5 * torch.sqrt(1 + 4 * epsilon_high / q_x)
clipped_ratio = torch.clamp(ratio, min=lower_bound, max=upper_bound)
surr1 = ratio * advantages
surr2 = clipped_ratio * advantages
obj = torch.min(surr1, surr2)
```

---

## 固定裁剪 (PPO-style Clipping)

当 `use_dynamic_clipping=False` 且 `clip_ratio > 0` 时，使用 PPO 风格的固定裁剪。

### 固定边界

$$\text{clip\_lower} = 1 - \epsilon$$
$$\text{clip\_upper} = 1 + \epsilon$$

其中 $\epsilon$ 是 `clip_ratio` 参数（例如 0.2 表示裁剪范围 [0.8, 1.2]）。

### 裁剪后的目标函数

$$\mathcal{L}_{\text{GRPO}}^{\text{PPO}} = -\frac{1}{N_{\text{tokens}}} \sum_{i \in \text{continuation}} \min\left(\text{ratio}_i \cdot A_i, \text{clipped\_ratio}_i \cdot A_i\right)$$

其中：

$$\text{clipped\_ratio}_i = \text{clip}(\text{ratio}_i, 1-\epsilon, 1+\epsilon)$$

代码实现：
```python
clip_lower = 1.0 - clip_ratio
clip_upper = 1.0 + clip_ratio
clipped_ratio = torch.clamp(ratio, min=clip_lower, max=clip_upper)
surr1 = ratio * advantages
surr2 = clipped_ratio * advantages
obj = torch.min(surr1, surr2)
```

### 无裁剪模式

当 `clip_ratio = 0.0` 时，不使用裁剪，直接使用原始 GRPO 目标：

$$\mathcal{L}_{\text{GRPO}}^{\text{no-clip}} = -\frac{1}{N_{\text{tokens}}} \sum_{i \in \text{continuation}} \text{ratio}_i \cdot A_i$$

或者等价地，直接使用策略梯度：

$$\mathcal{L}_{\text{GRPO}}^{\text{no-clip}} = -\frac{1}{N_{\text{tokens}}} \sum_{i \in \text{continuation}} \log \pi_\theta(x_i) \cdot A_i$$

---

## KL 散度惩罚

当 `use_kl_penalty=True` 时，添加 KL 散度正则化项，防止策略偏离参考模型太远。

### KL 散度计算

对于每个 token 位置，计算当前策略与参考策略之间的 KL 散度：

$$\text{KL}(p_\theta || p_{\text{ref}}) = \sum_{k} p_\theta(k) \cdot \left(\log p_\theta(k) - \log p_{\text{ref}}(k)\right)$$

其中：
- $p_\theta(k)$: 当前策略在词汇表位置 $k$ 的概率
- $p_{\text{ref}}(k)$: 参考策略在词汇表位置 $k$ 的概率

代码实现：
```python
log_p = torch.nn.functional.log_softmax(model_logits, dim=-1)
log_q = torch.nn.functional.log_softmax(ref_logits, dim=-1)
p = log_p.exp()
kl = (p * (log_p - log_q)).sum(dim=-1)
```

### KL 损失

只对 continuation tokens 计算 KL 损失：

$$\mathcal{L}_{\text{KL}} = \frac{1}{N_{\text{tokens}}} \sum_{i \in \text{continuation}} \text{KL}_i$$

---

## Prefix-SFT 损失

当 `enable_prefix=True` 时，对 prefix tokens 计算监督学习损失。

### Prefix-SFT 损失公式

$$\mathcal{L}_{\text{prefix-SFT}} = -\frac{1}{N_{\text{prefix}}} \sum_{i \in \text{prefix}} \log \pi_\theta(x_i)$$

其中 $N_{\text{prefix}}$ 是所有 prefix tokens 的总数。

代码实现：
```python
prefix_log_probs = log_probs * target_prefix_masks
prefix_sft_loss = -prefix_log_probs.sum() / (num_prefix_tokens + 1e-8)
```

---

## 奖励归一化

GRPO 的核心是**按问题分组归一化奖励**。对于每个问题 $q_i$，我们采样了 $M$ 个答案，然后计算这些答案的奖励均值和标准差，用于归一化。

### 标准 GRPO 归一化（按问题分组）

对于每个问题 $q_i$，计算其 $M$ 个答案的奖励均值和标准差：

$$\mu_i = \frac{1}{M} \sum_{j=1}^{M} r_{i,j}$$

$$\sigma_i = \sqrt{\frac{1}{M} \sum_{j=1}^{M} (r_{i,j} - \mu_i)^2}$$

然后对每个答案的奖励进行归一化，得到优势值：

$$A_{i,j} = \frac{r_{i,j} - \mu_i}{\sigma_i + \epsilon}$$

其中 $\epsilon$ 是一个小的常数（例如 $10^{-4}$），用于数值稳定性，防止除零。

**注意**：对于答案 $a_{i,j}$ 中的所有 token，优势值 $A_{i,j}[t]$ 都等于 $A_{i,j}$（即归一化的奖励）。

### 实现中的分组方式

在实际代码实现中，为了支持更灵活的分组策略，可能会按照以下方式分组：

1. **按 prefix 分组**：当使用 prefix 时，可以按照 prefix 对答案进行分组
2. **按长度分组**：当 `use_length_grouping=True` 时，按照生成序列的长度对奖励进行分组归一化

这些是实现的优化，但核心思想仍然是：**在同一组内的答案之间进行奖励归一化**。

---

## 总损失函数

最终的总损失函数是各个组件的加权和：

$$\mathcal{L}_{\text{total}} = \alpha_{\text{prefix}} \cdot \mathcal{L}_{\text{prefix-SFT}} + \mathcal{L}_{\text{GRPO}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

其中：
- $\alpha_{\text{prefix}}$: `prefix_sft_coeff`（默认 0.2）
- $\beta_{\text{KL}}$: `kl_coeff`（默认 0.01 或 0.02）

代码实现：
```python
grpo_loss = -(obj.sum() / num_target_tokens)
loss = prefix_sft_coeff * prefix_sft_loss + grpo_loss
if use_kl_penalty:
    loss += kl_coeff * kl_loss
```

### 损失组件说明

1. **Prefix-SFT Loss**: 只在启用 prefix 模式时计算，确保模型学习生成高质量的 prefix
2. **GRPO Loss**: 核心的策略优化损失，通过重要性采样和裁剪来稳定训练
3. **KL Loss**: 可选的正则化项，防止策略偏离参考模型太远

---

## 熵计算

虽然熵不直接参与损失计算，但用于监控训练过程：

$$\mathcal{H}(\pi_\theta) = -\sum_{k} p_\theta(k) \log p_\theta(k)$$

对于每个 continuation token：

$$\text{entropy}_i = \log \sum_{k} \exp(\text{logits}_i[k]) - \sum_{k} p_\theta(k) \cdot \text{logits}_i[k]$$

平均熵：

$$\bar{\mathcal{H}} = \frac{1}{N_{\text{tokens}}} \sum_{i \in \text{continuation}} \text{entropy}_i$$

---

## 梯度裁剪

为了防止梯度爆炸，使用梯度裁剪：

$$\text{grad} = \text{clip}(\nabla_\theta \mathcal{L}_{\text{total}}, -\text{max\_grad\_norm}, \text{max\_grad\_norm})$$

代码实现：
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
```

---

## 总结

GRPO 算法的核心思想是：

1. **重要性采样**: 使用旧策略的样本估计新策略的期望奖励
2. **裁剪机制**: 通过动态或固定裁剪限制策略更新幅度，保证训练稳定性
3. **正则化**: 通过 KL 惩罚和 Prefix-SFT 损失引导模型学习
4. **奖励归一化**: 通过分组归一化处理不同长度序列的奖励分布差异

这些机制共同作用，使得 GRPO 能够在保持训练稳定性的同时，有效地优化策略以获得更高的奖励。

