from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Episode:
    """Store all relevant information of an episode."""
    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]
    old_log_probs: List[float] = field(default_factory=list)  # Log probabilities from rollout (for importance sampling)
    # Prefix 训练模式相关字段
    prefix_source: str = "none"  # "deepseek", "3b", 或 "none"（未启用prefix模式）
    prefix_length: int = 0  # Prefix 的 token 长度（用于区分 prefix 和 continuation）
    prefix_old_log_probs: List[float] = field(default_factory=list)  # Prefix 的 log probs（用于 SFT loss）

@dataclass
class MiniBatch:
    """Batch of data for each training step."""
    # 通用字段（rollout/update_policy 需要）
    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]

    # 针对 GSM8K 的字段（用于取标准答案做奖励）
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    
    # Prefix 数据（如果启用 prefix 模式）
    prefix_data: List[Optional[Dict]] = field(default_factory=list)  # 每个问题的 prefix 数据

    # 兼容旧 countdown 任务的字段（如果不用可以为空）
    numbers: List[List[int]] = field(default_factory=list)
    target: List[int] = field(default_factory=list)