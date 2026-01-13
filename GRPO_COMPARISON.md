# GRPO 标准公式 vs 当前实现对比

本文档对比标准 GRPO 公式（方程3）与当前代码实现的差异。

## 标准 GRPO 公式（方程3）

### 原始标准公式（双重平均）

标准 GRPO 目标函数为：

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \left\{ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \cdot A_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \cdot A_{i,t} \right) \right] - \beta \cdot D_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}] \right\}$$

其中：
- $q$: 查询（问题）
- $o_i$: 第 $i$ 个观察（答案），$|o_i|$ 是其长度
- $G$: 每个问题的答案数量（组大小）
- $A_{i,t}$: 第 $i$ 个答案中第 $t$ 个 token 的优势
- $\epsilon$: PPO 裁剪参数
- $\beta$: KL 散度系数

### 关键特征

1. **双重平均结构**：
   - 外层：$\frac{1}{G} \sum_{i=1}^G$ - 对所有 $G$ 个答案求平均
   - 内层：$\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$ - 对每个答案内的所有 token 求平均
   - **这意味着先对每个答案内的 token 求平均，再对所有答案求平均**

2. **PPO 风格裁剪**：
   - 使用 `min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`
   - 裁剪范围：$[1-\epsilon, 1+\epsilon]$

3. **KL 散度正则化**：
   - $- \beta \cdot D_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}]$

---

## 全局平均形式的 GRPO 公式

### 改写后的公式（全局平均）

将标准公式改写为全局平均形式（对应当前代码实现）：

$$J_{\text{GRPO}}^{\text{global}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \left\{ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \left[ \min\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \cdot A_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \cdot A_{i,t} \right) \right] - \beta \cdot \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} D_{\text{KL}}[\pi_\theta(\cdot | q, o_{i,<t}) || \pi_{\text{ref}}(\cdot | q, o_{i,<t})] \right\}$$

### 改写说明

**主要变化**：

1. **平均方式改变**：
   - **原始**：$\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$ （双重平均）
   - **改写后**：$\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|}$ （全局平均）

2. **KL 散度也改为全局平均**：
   - **原始**：$- \beta \cdot D_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}]$ （整体 KL 散度）
   - **改写后**：$- \beta \cdot \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} D_{\text{KL}}[\pi_\theta(\cdot | q, o_{i,<t}) || \pi_{\text{ref}}(\cdot | q, o_{i,<t})]$ （每个 token 位置的 KL 散度求平均）

### 数学等价性

**当所有答案长度相等时**（$|o_i| = L$ 对所有 $i$）：
- 原始公式：$\frac{1}{G} \sum_{i=1}^G \frac{1}{L} \sum_{t=1}^{L} f_{i,t} = \frac{1}{G \cdot L} \sum_{i=1}^G \sum_{t=1}^{L} f_{i,t}$
- 改写后：$\frac{1}{G \cdot L} \sum_{i=1}^G \sum_{t=1}^{L} f_{i,t}$
- ✅ **完全等价**

**当答案长度不等时**：
- 原始公式：每个答案权重相等（$\frac{1}{G}$），无论长度
- 改写后：每个 token 权重相等（$\frac{1}{\sum_{i=1}^G |o_i|}$），长答案贡献更多
- ⚠️ **不等价**：长答案在全局平均中权重更大

### 符号说明

- $N_{\text{total}} = \sum_{i=1}^G |o_i|$：所有答案的总 token 数
- 其他符号与标准公式相同

---

## 当前代码实现

### 1. 奖励归一化

**代码位置**：`grpo.py:281-322` (`normalize_rewards_per_group`)

```python
# 按 prefix（或 prefix + 长度）分组
groups[key].append(episode)

# 对每个组内的奖励归一化
for group in groups.values():
    group_rewards = [item.reward for item in group]
    mean_reward = np.mean(group_rewards)
    std_reward = np.std(group_rewards)
    for episode in group:
        normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
```

**差异**：
- ✅ **一致**：按组归一化奖励（标准 GRPO 的核心思想）
- ✅ **一致**：按问题分组（用户确认代码中确实是按问题分组的）

### 2. 损失计算

**代码位置**：`grpo.py:604`

```python
grpo_loss = -(obj.sum() / num_target_tokens)
```

其中 `obj` 是每个 token 的 objective 值（已应用 PPO 裁剪）。

**差异**：
- ✅ **一致**：当前实现使用**全局平均**：$\frac{1}{N_{\text{total}}} \sum_{\text{all tokens}}$，对应改写后的全局平均公式
- 📝 **原始标准公式**：**双重平均**：$\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$

**数学等价性分析**：

标准公式的双重平均：
$$\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} f_{i,t} = \frac{1}{G} \sum_{i=1}^G \bar{f}_i$$

其中 $\bar{f}_i = \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} f_{i,t}$ 是第 $i$ 个答案内的平均。

当前实现的全局平均：
$$\frac{1}{N_{\text{total}}} \sum_{\text{all tokens}} f_{i,t} = \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} f_{i,t}$$

**当所有答案长度相等时**（$|o_i| = L$ 对所有 $i$）：
- 标准公式：$\frac{1}{G} \sum_{i=1}^G \bar{f}_i = \frac{1}{G \cdot L} \sum_{i=1}^G \sum_{t=1}^{L} f_{i,t}$
- 当前实现：$\frac{1}{G \cdot L} \sum_{i=1}^G \sum_{t=1}^{L} f_{i,t}$
- ✅ **等价**

**当答案长度不等时**：
- 标准公式：每个答案的权重相等（$\frac{1}{G}$），无论长度
- 当前实现：每个 token 的权重相等（$\frac{1}{N_{\text{total}}}$），长答案贡献更多
- ⚠️ **不等价**：长答案在当前实现中权重更大

### 3. PPO 裁剪

**代码位置**：`grpo.py:558-560` (动态裁剪) 或 `grpo.py:585-587` (固定裁剪)

```python
surr1 = ratio * advantages
surr2 = clipped_ratio * advantages
obj = torch.min(surr1, surr2)
```

**差异**：
- ✅ **一致**：使用 `min(ratio * A, clipped_ratio * A)`，与标准公式一致

### 4. KL 散度正则化

**代码位置**：`grpo.py:500-503, 610-611`

```python
if use_kl_penalty:
    kl_loss = (kl * target_continuation_masks).sum() / num_target_tokens
    loss += kl_coeff * kl_loss
```

**差异**：
- ✅ **一致**：支持 KL 散度正则化（可选）
- ⚠️ **实现差异**：KL 散度也是全局平均，而非按答案平均

---

## 总结

### ✅ 一致的部分

1. **奖励归一化**：按组归一化（核心思想一致）
2. **PPO 裁剪**：使用 `min(ratio * A, clipped_ratio * A)`
3. **KL 散度**：支持 KL 散度正则化（可选）

### ⚠️ 差异的部分

1. **平均方式**：
   - **标准公式**：双重平均（先按答案内平均，再按答案间平均）
   - **当前实现**：全局平均（所有 token 等权重）
   - **影响**：当答案长度不等时，长答案在当前实现中权重更大

2. **分组方式**：
   - ✅ **一致**：按问题分组（每个问题 $q$ 有 $G$ 个答案）

### 💡 建议

如果要完全对齐标准公式，需要修改：

1. **双重平均**：
   ```python
   # 当前
   grpo_loss = -(obj.sum() / num_target_tokens)
   
   # 标准公式（伪代码）
   # 按答案分组，先对每个答案内的 token 求平均
   per_answer_obj = []
   for answer_group in grouped_by_question:
       answer_obj = obj[answer_group].mean()  # 答案内平均
       per_answer_obj.append(answer_obj)
   grpo_loss = -(sum(per_answer_obj) / len(per_answer_obj))  # 答案间平均
   ```

2. **按问题分组**：
   - 需要跟踪每个 episode 对应的问题
   - 按问题 ID 分组，而非按 prefix 分组

---

## 实际影响评估

**当前实现的差异可能的影响**：

1. **长答案权重更大**：如果某些问题的答案更长，这些答案在梯度更新中会有更大的影响
2. **训练稳定性**：全局平均可能在某些情况下更稳定（因为所有 token 等权重）
3. **性能**：差异可能很小，特别是当答案长度分布相对均匀时

**结论**：当前实现对应**全局平均形式的 GRPO 公式**，核心思想（按组归一化奖励）保持一致。与原始双重平均公式的主要区别是：当答案长度不等时，长答案在全局平均中权重更大。这在某些场景下可能是有益的（例如，长答案通常包含更多信息），但也可能影响训练稳定性。

