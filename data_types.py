from dataclasses import dataclass, field
from typing import Dict, List

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

    # 兼容旧 countdown 任务的字段（如果不用可以为空）
    numbers: List[List[int]] = field(default_factory=list)
    target: List[int] = field(default_factory=list)