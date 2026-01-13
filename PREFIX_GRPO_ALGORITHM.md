# GRPO with Prefix 训练算法伪代码

## 算法概述

带 Prefix 的 GRPO 算法在标准 GRPO 基础上，增加了对 prefix tokens 的监督学习（SFT），使得模型能够学习生成高质量的推理前缀。

## 完整算法流程

```python
# ============================================
# 阶段 1: Rollout（采样阶段）
# ============================================

def rollout(model, batch, enable_prefix, prefix_dropout_prob):
    """
    对每个问题采样多个答案，可能包含 prefix
    """
    episodes = []
    
    for each question q_i in batch:
        # Step 1.1: 采样 Prefix（如果启用）
        if enable_prefix:
            # 随机决定是否使用 prefix（prefix_use_prob 控制）
            if random() < prefix_use_prob and has_prefix_data:
                # 随机选择使用 DeepSeek prefix 或 3B prefix
                if random() < prefix_dropout_prob:
                    prefix = sample_from_deepseek_prefixes(q_i)
                    prefix_source = "deepseek"
                    prefix_old_log_probs = []  # DeepSeek prefix 不需要 log probs
                else:
                    prefix = sample_from_3b_prefixes(q_i)
                    prefix_source = "3b"
                    prefix_old_log_probs = compute_old_log_probs(prefix)
            else:
                prefix = None
                prefix_source = "none"
                prefix_old_log_probs = []
        else:
            prefix = None
            prefix_source = "none"
            prefix_old_log_probs = []
        
        # Step 1.2: 对每个问题采样 M 个答案
        for j in range(M):  # M = num_answer_per_question
            # 生成 continuation（从 prefix 之后开始）
            if prefix:
                full_input = prompt + prefix
            else:
                full_input = prompt
            
            # 使用当前策略模型生成 continuation
            continuation = model.generate(
                input=full_input,
                max_len=max_gen_len,
                temperature=temperature
            )
            
            # 记录 continuation 的 old_log_probs（用于重要性采样）
            continuation_old_log_probs = compute_log_probs(model, continuation)
            
            # 计算完整输出的 reward
            full_output = prefix + continuation if prefix else continuation
            reward = reward_function(full_output, question=q_i, answer=gold_answer)
            
            # 创建 episode
            episode = Episode(
                prefix=prefix,
                continuation=continuation,
                prefix_old_log_probs=prefix_old_log_probs,
                continuation_old_log_probs=continuation_old_log_probs,
                reward=reward,
                prefix_length=len(prefix) if prefix else 0
            )
            episodes.append(episode)
    
    return episodes


# ============================================
# 阶段 2: Reward 归一化
# ============================================

def normalize_rewards(episodes, use_length_grouping):
    """
    按组归一化奖励，得到优势值
    """
    # 按 prefix（或 prefix + 长度）分组
    groups = group_episodes(episodes, use_length_grouping)
    
    for group in groups:
        # 计算组内奖励的均值和标准差
        group_rewards = [ep.reward for ep in group]
        μ = mean(group_rewards)
        σ = std(group_rewards)
        
        # 归一化每个 episode 的 reward（得到优势值）
        for episode in group:
            episode.advantage = (episode.reward - μ) / (σ + ε)
            # 注意：同一个 episode 中所有 token 的 advantage 都相同
    
    return episodes


# ============================================
# 阶段 3: 策略更新（梯度计算与更新）
# ============================================

def update_policy(model, episodes, enable_prefix, prefix_sft_coeff):
    """
    核心梯度更新函数
    """
    # Step 3.1: 归一化奖励
    episodes = normalize_rewards(episodes)
    
    # Step 3.2: 分批处理（micro-batching）
    for micro_batch in split_into_batches(episodes, micro_batch_size):
        
        # Step 3.3: 准备输入数据
        # 构建完整的 token 序列：[prompt] + [prefix] + [continuation]
        batch_token_ids = []
        batch_prefix_masks = []      # 标记哪些是 prefix tokens
        batch_continuation_masks = [] # 标记哪些是 continuation tokens
        batch_old_log_probs = []      # 旧策略的 log probs
        batch_advantages = []         # 归一化的优势值
        
        for episode in micro_batch:
            full_sequence = episode.prompt + episode.prefix + episode.continuation
            batch_token_ids.append(full_sequence)
            
            # 创建 mask
            prefix_mask = [0] * len(prompt) + [1] * len(prefix) + [0] * len(continuation)
            continuation_mask = [0] * len(prompt) + [0] * len(prefix) + [1] * len(continuation)
            
            batch_prefix_masks.append(prefix_mask)
            batch_continuation_masks.append(continuation_mask)
            
            # 组合 old_log_probs
            old_log_probs = [0] * len(prompt) + episode.prefix_old_log_probs + episode.continuation_old_log_probs
            batch_old_log_probs.append(old_log_probs)
            
            batch_advantages.append(episode.advantage)
        
        # Step 3.4: 前向传播
        input_ids = batch_token_ids[:, :-1]  # 输入（去掉最后一个）
        target_ids = batch_token_ids[:, 1:]  # 目标（去掉第一个）
        
        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
        
        # 计算当前策略的 log probs
        log_probs = compute_log_probs(logits, target_ids)  # (batch_size, seq_len)
        
        # Step 3.5: 计算 Prefix-SFT Loss（监督学习）
        # 只对 prefix tokens 计算
        prefix_sft_loss = 0.0
        if enable_prefix and num_prefix_tokens > 0:
            # 提取 prefix 位置的 log probs
            prefix_log_probs = log_probs * prefix_mask  # 只保留 prefix 位置
            
            # Prefix-SFT Loss: 最大化 prefix tokens 的对数概率
            prefix_sft_loss = -mean(prefix_log_probs)
            # 等价于：L_prefix = -1/N_prefix * sum(log π_θ(prefix_token))
        
        # Step 3.6: 计算 GRPO Loss（强化学习）
        # 只对 continuation tokens 计算
        
        # 3.6.1: 计算重要性采样比率
        old_log_probs = batch_old_log_probs[:, 1:]  # 对齐到 target 位置
        ratio = exp(log_probs - old_log_probs)  # (batch_size, seq_len)
        
        # 3.6.2: PPO 裁剪（可选）
        if use_clipping:
            clipped_ratio = clip(ratio, 1-ε, 1+ε)
            obj = min(ratio * advantages, clipped_ratio * advantages)
        else:
            obj = ratio * advantages
        
        # 3.6.3: 只对 continuation tokens 计算
        obj = obj * continuation_mask  # 屏蔽非 continuation 位置
        
        # 3.6.4: 计算 GRPO Loss（全局平均）
        num_continuation_tokens = sum(continuation_mask)
        grpo_loss = -sum(obj) / num_continuation_tokens
        # 等价于：L_GRPO = -1/N_cont * sum(ratio * A for continuation tokens)
        
        # Step 3.7: 计算 KL Loss（可选，只对 continuation）
        kl_loss = 0.0
        if use_kl_penalty:
            ref_logits = ref_model(input_ids)  # 参考模型的 logits
            kl = compute_kl_divergence(logits, ref_logits)
            kl_loss = mean(kl * continuation_mask)
        
        # Step 3.8: 组合总 Loss
        total_loss = (
            prefix_sft_coeff * prefix_sft_loss +  # Prefix 监督学习
            grpo_loss +                           # GRPO 强化学习
            kl_coeff * kl_loss                    # KL 正则化（可选）
        )
        
        # Step 3.9: 反向传播（梯度计算）
        total_loss.backward()
        # 这会计算：
        #   ∇_θ L_total = prefix_sft_coeff * ∇_θ L_prefix 
        #                + ∇_θ L_GRPO 
        #                + kl_coeff * ∇_θ L_KL
    
    # Step 3.10: 梯度裁剪和更新
    grad_norm = clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()  # 更新参数
    optimizer.zero_grad()


# ============================================
# 梯度更新详解
# ============================================

"""
梯度更新的数学公式：

总损失函数：
    L_total = α_prefix * L_prefix + L_GRPO + β_KL * L_KL

其中：

1. Prefix-SFT Loss（监督学习）：
    L_prefix = -1/N_prefix * Σ_{t ∈ prefix} log π_θ(token_t | context)
    
   梯度：
    ∇_θ L_prefix = -1/N_prefix * Σ_{t ∈ prefix} ∇_θ log π_θ(token_t | context)
   
   作用：让模型学习生成高质量的 prefix（通过监督学习）

2. GRPO Loss（强化学习）：
    L_GRPO = -1/N_cont * Σ_{t ∈ continuation} ratio_t * A_t
    
   其中：
   - ratio_t = π_θ(token_t) / π_old(token_t)  # 重要性采样比率
   - A_t = (reward - μ) / σ  # 归一化的优势值
   
   梯度：
    ∇_θ L_GRPO = -1/N_cont * Σ_{t ∈ continuation} ∇_θ (ratio_t * A_t)
                = -1/N_cont * Σ_{t ∈ continuation} A_t * ∇_θ log π_θ(token_t)
   
   作用：通过策略梯度优化 continuation 的生成（通过强化学习）

3. KL Loss（正则化，可选）：
    L_KL = 1/N_cont * Σ_{t ∈ continuation} KL(π_θ || π_ref)
    
   梯度：
    ∇_θ L_KL = 1/N_cont * Σ_{t ∈ continuation} ∇_θ KL(π_θ || π_ref)
   
   作用：防止策略偏离参考模型太远

最终梯度：
    ∇_θ L_total = α_prefix * ∇_θ L_prefix + ∇_θ L_GRPO + β_KL * ∇_θ L_KL

参数更新：
    θ ← θ - lr * ∇_θ L_total
"""


# ============================================
# 关键设计点
# ============================================

"""
1. **Prefix 和 Continuation 的分离**：
   - Prefix tokens：使用监督学习（SFT），最大化对数概率
   - Continuation tokens：使用强化学习（GRPO），通过优势值优化

2. **梯度更新的独立性**：
   - Prefix-SFT loss 只影响 prefix 位置的梯度
   - GRPO loss 只影响 continuation 位置的梯度
   - 两者通过加权组合：L_total = α * L_prefix + L_GRPO

3. **Prefix 的来源**：
   - DeepSeek prefix：高质量，但不需要 old_log_probs（因为是外部生成）
   - 3B prefix：模型自己生成的，需要 old_log_probs（用于重要性采样）
   - 无 prefix：prefix_length = 0，不计算 prefix_sft_loss

4. **优势值的计算**：
   - 同一个 episode 的所有 token（包括 prefix 和 continuation）共享相同的优势值
   - 但只有 continuation tokens 参与 GRPO loss 计算
   - Prefix tokens 不参与 GRPO，只参与 SFT

5. **梯度更新的流程**：
   - 每个 micro batch 计算一次 loss 并 backward（梯度累积）
   - 所有 batch 处理完后，统一进行梯度裁剪和参数更新
   - 这样可以在内存受限时使用较小的 micro_batch_size
"""


# ============================================
# 完整训练循环
# ============================================

def train_loop(model, dataset, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataset:
            # 1. Rollout: 采样答案（可能包含 prefix）
            episodes = rollout(
                model=model,
                batch=batch,
                enable_prefix=True,
                prefix_dropout_prob=0.5
            )
            
            # 2. Update: 计算梯度并更新参数
            results = update_policy(
                model=model,
                episodes=episodes,
                enable_prefix=True,
                prefix_sft_coeff=0.2
            )
            
            # 3. 记录指标
            log_metrics(results)
```


