import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

USER_TEMPLATE = (
    "Solve the following math word problem. "
    "Show your reasoning inside <think></think> tags "
    "and put the final numeric answer inside <answer></answer> tags.\n\n"
    "Problem: {question}"
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

class GSM8KDataset(Dataset):
    """Unified dataset for GSM8K main and socratic splits."""
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
        config_name: str = "main",
    ):
        base_dir = Path(data_path) / config_name
        split_file = base_dir / f"{split}.parquet"

        explicit_split_file = False
        if split_file.exists():
            # 直接读取指定 split 文件，例如 .../gsm8k/main/train.parquet
            data = pd.read_parquet(split_file)
            explicit_split_file = True
        else:
            # 尝试读取分片文件，例如 train-00000-of-00001.parquet
            shard_files = sorted(base_dir.glob(f"{split}-*.parquet"))
            if shard_files:
                data = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
                explicit_split_file = True
            else:
                # 找不到就报错，提示应提供 {split}.parquet
                raise FileNotFoundError(
                    f"Cannot find parquet for split='{split}'. "
                    f"Tried: {split_file} and shards {base_dir}/{split}-*.parquet"
                )

        # 如果是明确的 split 文件，就不再做 test_size 切分；否则（保留旧逻辑）才切分
        if explicit_split_file:
            self.data = data
        else:
            self.data = data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        q = item["question"]
        a = item["answer"]
        item.update(self.encode_prefix(q))
        return item

    def encode_prefix(self, question: str):
        user_message = USER_TEMPLATE.format(question=question)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch):
        return MiniBatch(
            questions=[x["question"] for x in batch],
            answers=[x["answer"] for x in batch],
            prefix=[x["prefix"] for x in batch],
            prefix_tokens=[x["prefix_tokens"] for x in batch],
            prefix_token_ids=[x["prefix_token_ids"] for x in batch],
        )

def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function_gsm8k(response: str, gold_answer: str) -> float:
    """
    Extracts the predicted answer from <answer>...</answer>
    and compares with the GSM8K official final answer format "#### 23".
    """

    # --- 1. extract predicted answer inside <answer>...</answer> ---
    answer_match = re.search(r"<answer>(.*?)<\/answer>", response, re.DOTALL)
    if not answer_match:
        return 0.0

    pred_text = answer_match.group(1).strip()

    # Extract only numbers from prediction
    pred_nums = re.findall(r"-?\d+\.?\d*", pred_text)
    if not pred_nums:
        return 0.0

    try:
        pred = float(pred_nums[-1])   # use last number user wrote
    except:
        return 0.0

    # --- 2. extract official gold answer from "#### 23" ---
    gold_match = re.search(r"####\s*(-?\d+\.?\d*)", gold_answer)
    if not gold_match:
        return 0.0

    try:
        gold = float(gold_match.group(1))
    except:
        return 0.0

    # compare numeric equality
    if abs(pred - gold) < 1e-6:
        return 1.0
    
    return 0.0


def reward_function(response: str, question=None, answer=None, end_token=None):
    """
    reward = 0.1 * format_reward + answer_reward
    where answer_reward is correctness of the GSM8K final answer
    """

    # keep your format reward unchanged
    format_reward = format_reward_function("<think>" + response, end_token)

    # GSM8K correctness reward
    answer_reward = answer_reward_function_gsm8k(response, gold_answer=answer)

    return {
        "reward": 0.1 * format_reward + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }