import dataclasses
import gc
import math
import random
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer

# 同一道题采样K个答案
@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
    enable_prefix: bool = False,
    prefix_dropout_prob: float = 0.5,
    prefix_use_prob: float = 1.0,  # 训练时使用prefix的概率（用于与eval分布一致）
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Step 1: Prefix 采样（如果启用）
    prefix_info_list = []  # 存储每个问题的 prefix 信息
    if enable_prefix:
        # 调试信息：检查 batch 是否有 prefix_data
        if not hasattr(batch, 'prefix_data'):
            print(f"Warning: batch has no 'prefix_data' attribute, prefix mode disabled for this batch")
        elif not batch.prefix_data:
            print(f"Warning: batch.prefix_data is empty, prefix mode disabled for this batch")
        
        for i, question in enumerate(batch.questions):
            prefix_data = batch.prefix_data[i] if hasattr(batch, 'prefix_data') and i < len(batch.prefix_data) else None
            
            # 决定是否使用prefix（用于与eval分布一致）
            # prefix_use_prob=1.0 表示总是使用prefix（如果可用）
            # prefix_use_prob=0.5 表示50%的概率使用prefix，50%的概率不使用（模拟eval时的无prefix情况）
            has_prefix_data = prefix_data and (prefix_data.get("deepseek_prefixes") or prefix_data.get("3b_prefixes"))
            if has_prefix_data:
                use_prefix_this_sample = random.random() < prefix_use_prob
            else:
                use_prefix_this_sample = False
            
            # 调试信息（只打印前几个样本）
            # if i < 3 and enable_prefix:
                # print(f"Sample {i}: has_prefix_data={has_prefix_data}, prefix_use_prob={prefix_use_prob}, use_prefix_this_sample={use_prefix_this_sample}")
            
            if use_prefix_this_sample:
                # 从预生成的 prefix 中随机选择
                use_deepseek = random.random() < prefix_dropout_prob
                
                if use_deepseek and prefix_data.get("deepseek_prefixes"):
                    # 从 DeepSeek prefixes 中随机选择一个
                    deepseek_prefixes = prefix_data["deepseek_prefixes"]
                    if deepseek_prefixes:
                        selected = random.choice(deepseek_prefixes)
                        prefix_text = selected["text"]
                        prefix_token_ids = selected["token_ids"]
                        prefix_tokens = selected["tokens"]
                        prefix_source = "deepseek"
                        prefix_old_log_probs = []  # DeepSeek 的 prefix 不需要 log probs
                    else:
                        # 如果没有 DeepSeek prefix，使用 3B
                        if prefix_data.get("3b_prefixes"):
                            selected = random.choice(prefix_data["3b_prefixes"])
                            prefix_text = selected["text"]
                            prefix_token_ids = selected["token_ids"]
                            prefix_tokens = selected["tokens"]
                            prefix_old_log_probs = selected.get("log_probs", [])
                            prefix_source = "3b"
                        else:
                            # 如果都没有，使用空 prefix
                            prefix_text = ""
                            prefix_token_ids = []
                            prefix_tokens = []
                            prefix_source = "none"
                            prefix_old_log_probs = []
                else:
                    # 使用 3B prefix
                    if prefix_data.get("3b_prefixes"):
                        selected = random.choice(prefix_data["3b_prefixes"])
                        prefix_text = selected["text"]
                        prefix_token_ids = selected["token_ids"]
                        prefix_tokens = selected["tokens"]
                        prefix_old_log_probs = selected.get("log_probs", [])
                        prefix_source = "3b"
                    elif prefix_data.get("deepseek_prefixes"):
                        # 如果没有 3B prefix，使用 DeepSeek
                        selected = random.choice(prefix_data["deepseek_prefixes"])
                        prefix_text = selected["text"]
                        prefix_token_ids = selected["token_ids"]
                        prefix_tokens = selected["tokens"]
                        prefix_source = "deepseek"
                        prefix_old_log_probs = []
                    else:
                        # 如果都没有，使用空 prefix
                        prefix_text = ""
                        prefix_token_ids = []
                        prefix_tokens = []
                        prefix_source = "none"
                        prefix_old_log_probs = []
            else:
                # 不使用prefix（可能是prefix_data为None，或者被prefix_use_prob随机dropout）
                prefix_text = ""
                prefix_token_ids = []
                prefix_tokens = []
                prefix_source = "none"
                prefix_old_log_probs = []
            
            prefix_info_list.append({
                "prefix_text": prefix_text,
                "prefix_token_ids": prefix_token_ids,
                "prefix_tokens": prefix_tokens,
                "prefix_source": prefix_source,
                "prefix_old_log_probs": prefix_old_log_probs,
            })
    else:
        # 未启用 prefix 模式，使用空 prefix
        for i in range(len(batch.questions)):
            prefix_info_list.append({
                "prefix_text": "",
                "prefix_token_ids": [],
                "prefix_tokens": [],
                "prefix_source": "none",
                "prefix_old_log_probs": [],
            })
    
    # 构建完整的 prefix（原始 prompt + 生成的 prefix）
    full_prefix_token_ids = []
    for i, original_prefix_ids in enumerate(batch.prefix_token_ids):
        # 将生成的 prefix token ids 添加到原始 prompt 后面
        full_prefix = original_prefix_ids + prefix_info_list[i]["prefix_token_ids"]
        full_prefix_token_ids.append(full_prefix)
    
    prefix_token_ids = full_prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    token_log_probs = torch.zeros((bsz, total_len), dtype=torch.float, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        
        # Compute log probabilities WITHOUT temperature (for importance sampling)
        log_probs_all = torch.log_softmax(logits[:, -1], dim=-1)
        
        # Apply temperature scaling ONLY for sampling
        scaled_logits = logits[:, -1] / temperature
        scaled_log_probs = torch.log_softmax(scaled_logits, dim=-1)
        probs = torch.exp(scaled_log_probs)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        
        # Store log probabilities WITHOUT temperature (for importance sampling ratio)
        selected_log_probs = log_probs_all.gather(1, next_token.unsqueeze(1)).squeeze(1)
        token_log_probs[:, cur_pos] = selected_log_probs
        
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()
    token_log_probs_list = token_log_probs.tolist()

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        # 获取原始 prompt 长度和 prefix 长度
        original_prompt_len = len(batch.prefix_token_ids[i])
        prefix_len = len(prefix_info_list[i]["prefix_token_ids"])
        prefix_start_pos = original_prompt_len
        continuation_start_pos = original_prompt_len + prefix_len
        
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            
            # Continuation token ids（从 continuation_start_pos 开始）
            continuation_token_ids = tokens_list[idx][continuation_start_pos:]
            continuation_log_probs = token_log_probs_list[idx][continuation_start_pos:]
            
            # remove padding tokens
            if pad_token_id in continuation_token_ids:
                pad_idx = continuation_token_ids.index(pad_token_id)
                continuation_token_ids = continuation_token_ids[:pad_idx]
                continuation_log_probs = continuation_log_probs[:pad_idx]
            
            # 完整输出（prefix + continuation）用于 reward 计算
            # 注意：原始 prompt 已经包含 <think>，所以：
            # - prefix 在 <think> 后面
            # - continuation 包含后续推理 + </think> + <answer>答案</answer>
            # 最终格式：<think>prefix + continuation
            full_output_token_ids = prefix_info_list[i]["prefix_token_ids"] + continuation_token_ids
            full_output_text = tokenizer.detokenize(full_output_token_ids)
            
            # 计算 reward（基于完整输出）
            # reward_function 会在 response 前面加上 <think>，所以最终是：
            # <think> + (prefix + continuation)
            rewards = reward_function(
                response=full_output_text,
                question=batch.questions[i],
                answer=batch.answers[i],
                end_token=end_token,
            )
            
            # 构建完整的 prefix（原始 prompt + 生成的 prefix）
            full_prefix_text = batch.prefix[i] + prefix_info_list[i]["prefix_text"]
            full_prefix_token_ids_combined = batch.prefix_token_ids[i] + prefix_info_list[i]["prefix_token_ids"]
            full_prefix_tokens_combined = batch.prefix_tokens[i] + prefix_info_list[i]["prefix_tokens"]
            
            episode = Episode(
                prefix=full_prefix_text,
                text=full_prefix_text + tokenizer.detokenize(continuation_token_ids),
                prefix_token_ids=full_prefix_token_ids_combined,
                prefix_tokens=full_prefix_tokens_combined,
                generated_token_ids=continuation_token_ids,  # 只包含 continuation
                old_log_probs=continuation_log_probs,  # 只包含 continuation 的 log probs
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
                prefix_source=prefix_info_list[i]["prefix_source"],
                prefix_length=prefix_len,
                prefix_old_log_probs=prefix_info_list[i]["prefix_old_log_probs"],
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode], use_length_grouping: bool = False) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix (and optionally response length bucket)."""
    groups = defaultdict(list)
    
    if use_length_grouping:
        # Compute length distribution statistics
        lengths = [len(episode.generated_token_ids) for episode in episodes]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        # Handle edge case where all lengths are the same (std = 0)
        if std_length < 1e-6:
            std_length = 1.0  # Use a small default std to avoid division issues
        # Group by: < mean - sigma (short), mean - sigma to mean + sigma (medium), > mean + sigma (long)
        lower_bound = mean_length - std_length
        upper_bound = mean_length + std_length
    
    for episode in episodes:
        if use_length_grouping:
            # Determine length bucket based on normal distribution
            length = len(episode.generated_token_ids)
            if length < lower_bound:
                bucket = "short"
            elif length <= upper_bound:
                bucket = "medium"
            else:
                bucket = "long"
            key = (tuple(episode.prefix), bucket)
        else:
            key = tuple(episode.prefix)
        
        groups[key].append(episode)
        
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy

def compute_kl_loss(model_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(p || q)
    """
    log_p = torch.nn.functional.log_softmax(model_logits, dim=-1)   # (B, L, K)
    log_q = torch.nn.functional.log_softmax(ref_logits,  dim=-1)    # (B, L, K)
    p = log_p.exp()    # (B, L, K)
    # standard KL
    kl = (p * (log_p - log_q)).sum(dim=-1)   # (B, L)
    return kl

def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    ref_model=None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    kl_coeff: float = 0.0,
    use_length_grouping: bool = False,
    use_dynamic_clipping: bool = True,
    use_kl_penalty: bool = False,
    clip_ratio: float = 0.2,
    enable_prefix: bool = False,
    prefix_sft_coeff: float = 0.2,
):
    """Update the policy using the GRPO algorithm.
    
    Args:
        model: The policy model to update
        optimizer: Optimizer for the model
        episodes: List of episodes from rollout
        micro_batch_size: Size of micro batches
        pad_token_id: Padding token ID
        max_grad_norm: Maximum gradient norm for clipping
        device: Device to run on
        dtype: Data type
        ref_model: Reference model for KL penalty (optional, required if use_kl_penalty=True)
        epsilon_low: Lower bound epsilon for dynamic clipping
        epsilon_high: Upper bound epsilon for dynamic clipping
        kl_coeff: Coefficient for KL penalty (only used if use_kl_penalty=True)
        use_length_grouping: Whether to use length-based grouping for reward normalization
        use_dynamic_clipping: Whether to use dynamic clipping
        use_kl_penalty: Whether to use KL penalty
        clip_ratio: Fixed clip ratio for PPO-style clipping (when use_dynamic_clipping=False)
    """
    episodes = normalize_rewards_per_group(episodes, use_length_grouping=use_length_grouping)
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    num_prefix_tokens = sum(episode.prefix_length for episode in episodes) if enable_prefix else 0
    entropy = 0.0
    prefix_sft_loss_total = 0.0
    total_clip_stats = {
        "clipped_lower": 0,
        "clipped_upper": 0,
        "total": 0,
        "ratio_sum": 0.0,
        "clipped_ratio_sum": 0.0,
    }

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # 构建 batch_old_log_probs：prefix 的 old_log_probs + continuation 的 old_log_probs
        batch_old_log_probs = []
        batch_prefix_masks = []  # Prefix mask（用于 SFT loss）
        batch_continuation_masks = []  # Continuation mask（用于 GRPO loss）
        for i, episode in enumerate(batch_episodes):
            prefix_len = episode.prefix_length if enable_prefix else 0
            original_prompt_len = len(episode.prefix_token_ids) - prefix_len
            
            # Prefix old_log_probs（如果有）
            if enable_prefix and episode.prefix_old_log_probs:
                prefix_old_log_probs = episode.prefix_old_log_probs
            else:
                prefix_old_log_probs = [0.0] * prefix_len
            
            # 完整的 old_log_probs：原始 prompt (0) + prefix + continuation
            full_old_log_probs = (
                [0.0] * original_prompt_len
                + prefix_old_log_probs
                + episode.old_log_probs
                + [0.0] * (batch_max_length - batch_lengths[i])
            )
            batch_old_log_probs.append(full_old_log_probs)
            
            # Prefix mask：只对 prefix token 为 1（不包括原始 prompt）
            prefix_mask = (
                [0] * original_prompt_len
                + [1] * prefix_len
                + [0] * (len(episode.generated_token_ids) + batch_max_length - batch_lengths[i])
            )
            batch_prefix_masks.append(prefix_mask)
            
            # Continuation mask：只对 continuation token 为 1
            continuation_mask = (
                [0] * (original_prompt_len + prefix_len)
                + [1] * len(episode.generated_token_ids)
                + [0] * (batch_max_length - batch_lengths[i])
            )
            batch_continuation_masks.append(continuation_mask)
        
        batch_masks = batch_continuation_masks  # 保持兼容性，但实际使用 continuation_mask
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_old_log_probs = torch.tensor(batch_old_log_probs, device=device, dtype=torch.float32)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_prefix_masks = torch.tensor(batch_prefix_masks, device=device, dtype=torch.bool) if enable_prefix else None
        batch_continuation_masks = torch.tensor(batch_continuation_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_continuation_masks = batch_continuation_masks[:, 1:]  # 只对 continuation 计算 GRPO
            target_prefix_masks = batch_prefix_masks[:, 1:] if enable_prefix and batch_prefix_masks is not None else None
            old_log_probs = batch_old_log_probs[:, 1:]
            logits = model.forward(input_token_ids).float()
            
            # Compute reference logits if KL penalty is enabled
            if use_kl_penalty:
                if ref_model is None:
                    raise ValueError("ref_model must be provided when use_kl_penalty=True")
                with torch.no_grad():
                    ref_logits = ref_model.forward(input_token_ids).float()
            else:
                ref_logits = None

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        # Step 5: Prefix-SFT Loss（如果启用 prefix 模式且有 prefix token）
        prefix_sft_loss = torch.tensor(0.0, device=device)
        if enable_prefix and target_prefix_masks is not None and num_prefix_tokens > 0:
            # 对 prefix token 计算 SFT loss（无论来源）
            # 注意：当不使用prefix时（prefix_length=0），num_prefix_tokens=0，这里不会执行
            prefix_log_probs = log_probs * target_prefix_masks
            prefix_sft_loss = -prefix_log_probs.sum() / (num_prefix_tokens + 1e-8)
            prefix_sft_loss_total += prefix_sft_loss.item()

        # Compute KL loss if KL penalty is enabled（只对 continuation）
        kl_loss = torch.tensor(0.0, device=device)
        if use_kl_penalty and ref_logits is not None:
            kl = compute_kl_loss(logits, ref_logits)
            kl_loss = (kl * target_continuation_masks).sum() / num_target_tokens  # 只对 continuation 计算 KL

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_continuation_masks).sum() / num_target_tokens  # 只对 continuation 计算 entropy

        # Compute objective based on selected method
        advantages = batch_advantages[:, None]
        
        clip_stats = {
            "clipped_lower": 0,
            "clipped_upper": 0,
            "total": 0,
            "ratio_sum": 0.0,
            "clipped_ratio_sum": 0.0,
        }
        
        # Step 6: Conditional GRPO Loss（只对 continuation token）
        # 使用原始 log_probs 和 old_log_probs 计算 ratio（不要先mask）
        # 这样在 mask=0 的位置 ratio 也会有正确的值（虽然我们不会使用）
        ratio = torch.exp(log_probs - old_log_probs)
        
        if use_dynamic_clipping:
            # Importance Sampling with Dynamic Adaptive Clipping（只对 continuation）
            # 使用原始 old_log_probs 计算 q_x（对所有位置）
            q_x = torch.exp(old_log_probs)
            
            # Dynamic bounds（对所有位置计算，但只在continuation位置使用）
            # L(x) = 0.5 + 0.5 * sqrt(max(1 - 4*eps_low/q(x), 0))
            val_low = 1 - 4 * epsilon_low / (q_x + 1e-10)
            val_low = torch.clamp(val_low, min=0.0)
            lower_bound = 0.5 + 0.5 * torch.sqrt(val_low)
            
            # U(x) = 0.5 + 0.5 * sqrt(1 + 4*eps_high/q(x))
            val_high = 1 + 4 * epsilon_high / (q_x + 1e-10)
            upper_bound = 0.5 + 0.5 * torch.sqrt(val_high)
            
            # Track clipping statistics（只统计 continuation tokens）
            continuation_ratio = ratio * target_continuation_masks
            # 只在continuation位置比较ratio和边界
            continuation_lower_bound = lower_bound * target_continuation_masks
            continuation_upper_bound = upper_bound * target_continuation_masks
            clipped_lower = ((continuation_ratio < continuation_lower_bound) & target_continuation_masks).sum().item()
            clipped_upper = ((continuation_ratio > continuation_upper_bound) & target_continuation_masks).sum().item()
            total_tokens = target_continuation_masks.sum().item()
            clip_stats = {
                "clipped_lower": clipped_lower,
                "clipped_upper": clipped_upper,
                "total": total_tokens
            }
            
            clipped_ratio = torch.clamp(ratio, min=lower_bound, max=upper_bound)
            clip_stats["ratio_sum"] = continuation_ratio.sum().item()
            clip_stats["clipped_ratio_sum"] = (clipped_ratio * target_continuation_masks).sum().item()
            
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            obj = torch.min(surr1, surr2)
            # 只对 continuation 计算 objective
            obj = obj * target_continuation_masks
        else:
            # PPO-style fixed clipping (only if clip_ratio > 0)
            if clip_ratio > 0.0:
                # ratio 已经在上面用原始 log_probs 和 old_log_probs 计算了
                clip_lower = 1.0 - clip_ratio
                clip_upper = 1.0 + clip_ratio
            
                # Track clipping statistics（只统计 continuation tokens）
                continuation_ratio = ratio * target_continuation_masks
                clipped_lower = (continuation_ratio < clip_lower).sum().item()
                clipped_upper = (continuation_ratio > clip_upper).sum().item()
                total_tokens = target_continuation_masks.sum().item()
                clip_stats = {
                    "clipped_lower": clipped_lower,
                    "clipped_upper": clipped_upper,
                    "total": total_tokens,
                    "ratio_sum": continuation_ratio.sum().item(),
                }
            
                clipped_ratio = torch.clamp(ratio, min=clip_lower, max=clip_upper)
                clip_stats["clipped_ratio_sum"] = (clipped_ratio * target_continuation_masks).sum().item()
            
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages
                obj = torch.min(surr1, surr2)
                # 只对 continuation 计算 objective
                obj = obj * target_continuation_masks
            else:
                # No clipping: use original GRPO objective（只对 continuation）
                continuation_log_probs = log_probs * target_continuation_masks
                obj = continuation_log_probs * advantages

        # Accumulate clip statistics (both for dynamic and fixed clipping)
        total_clip_stats["clipped_lower"] += clip_stats["clipped_lower"]
        total_clip_stats["clipped_upper"] += clip_stats["clipped_upper"]
        total_clip_stats["total"] += clip_stats["total"]
        total_clip_stats["ratio_sum"] += clip_stats["ratio_sum"]
        total_clip_stats["clipped_ratio_sum"] += clip_stats["clipped_ratio_sum"]

        # Step 8: 总 Loss
        # GRPO loss（只对 continuation）
        grpo_loss = -(obj.sum() / num_target_tokens)
        
        # 总 loss = prefix-SFT loss + GRPO loss + KL loss
        loss = prefix_sft_coeff * prefix_sft_loss + grpo_loss
        
        # Add KL penalty if enabled
        if use_kl_penalty:
            loss += kl_coeff * kl_loss
        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    result = {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
    
    if enable_prefix:
        result["prefix_sft_loss"] = prefix_sft_loss_total / num_micro_batches if num_micro_batches > 0 else 0.0
    
    if use_kl_penalty:
        result["kl_loss"] = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
    
    # Return clip statistics (both for dynamic and fixed clipping)
    total_tokens = total_clip_stats["total"]
    if total_tokens > 0:
        result["clip_frac_lower"] = total_clip_stats["clipped_lower"] / total_tokens
        result["clip_frac_upper"] = total_clip_stats["clipped_upper"] / total_tokens
        result["clip_frac_total"] = (total_clip_stats["clipped_lower"] + total_clip_stats["clipped_upper"]) / total_tokens
        result["mean_ratio"] = total_clip_stats["ratio_sum"] / total_tokens
        result["mean_clipped_ratio"] = total_clip_stats["clipped_ratio_sum"] / total_tokens
    else:
        result["clip_frac_lower"] = 0.0
        result["clip_frac_upper"] = 0.0
        result["clip_frac_total"] = 0.0
        result["mean_ratio"] = 0.0
        result["mean_clipped_ratio"] = 0.0
    
    return result
