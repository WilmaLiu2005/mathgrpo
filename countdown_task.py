import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer


SYSTEM_MESSAGE = (
    "You are a helpful assistant that solves math problems accurately."
)

USER_TEMPLATE = (
    "Solve the following math word problem.\n"
    "Provide your reasoning inside <think></think> tags, "
    "and put the final numeric answer inside <answer></answer> tags.\n\n"
    "Problem: {question}"
)

RESPONSE_PROMPT = "<think>"

# Prefix 训练模式的 prompt 模板（用于生成 prefix，只要求初始思考步骤）
PREFIX_USER_TEMPLATE = (
    "Think about the following math problem.\n"
    "Only provide the first 1-2 sentences of your initial thinking.\n"
    "Do NOT solve it completely or give the final answer.\n"
    "Do NOT output anything else after these 1-2 sentences.\n"
    "Do NOT include any auxiliary labels like 'think:', 'reasoning:', or similar prefixes.\n"
    "Just output the thinking sentences directly.\n"
    "Stop immediately after the second sentence.\n\n"
    "Problem: {question}"
)


def clean_prefix_text(text: str) -> str:
    """
    清理 prefix 文本，移除特殊 token 和乱码。
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    # 1. 移除特殊 token（如 <|endoftext|>, <|im_start|>, <|im_end|> 等）
    special_tokens = [
        r'<\|endoftext\|>',
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'<\|.*?\|>',  # 匹配所有 <|...|> 格式的特殊 token
    ]
    for pattern in special_tokens:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 2. 移除控制字符和不可打印字符（保留空格、换行、标点）
    # 只保留 ASCII 可打印字符、常见标点和空格
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    # 3. 移除乱码字符（非 ASCII 字母、数字、常见标点）
    # 保留：英文字母、数字、常见标点、空格
    cleaned = []
    for char in text:
        # 保留 ASCII 字母、数字、常见标点、空格
        if (char.isascii() and (char.isalnum() or char in ' .,!?;:()[]{}"\'-')) or char.isspace():
            cleaned.append(char)
    
    text = ''.join(cleaned)
    
    # 4. 移除辅助标签（如 "think:", "reasoning:", "thought:", 等）
    # 匹配行首的标签（不区分大小写），后面可能跟着冒号或空格
    auxiliary_patterns = [
        r'^\s*(think|reasoning|thought|thinking|step|process|analysis|approach):\s*',
        r'^\s*(think|reasoning|thought|thinking|step|process|analysis|approach)\s+',
    ]
    for pattern in auxiliary_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 5. 清理多余的空格
    text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
    text = text.strip()
    
    return text


def truncate_by_sentences(text: str, max_sentences: int = 2) -> str:
    """
    按句号截断文本，保留前 max_sentences 句。
    
    Args:
        text: 输入文本
        max_sentences: 最多保留的句子数（默认2句）
    
    Returns:
        截断后的文本（包含句号）
    """
    import re
    
    # 匹配句号（包括英文句号、中文句号、问号、感叹号等）
    # 注意：只有句号后面跟着空格、换行或文本结束才认为是句点
    # 对于英文句号，必须后面有空格（避免匹配小数点等）
    sentence_end_pattern = r'(?:[。!?]|\.\s+)(?:\s|$)'
    
    # 找到所有句号位置
    matches = list(re.finditer(sentence_end_pattern, text))
    
    if len(matches) >= max_sentences:
        # 找到第 max_sentences 个句号，截断到该句号之后
        end_pos = matches[max_sentences - 1].end()
        return text[:end_pos].strip()
    else:
        # 如果句号数量不足，返回全部文本
        return text.strip()

class GSM8KDataset(Dataset):
    """Unified dataset for GSM8K main and socratic splits."""
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
        config_name: str = "main",
        prefix_file: Optional[str] = None,  # Prefix 文件路径（JSON格式）
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
        
        # 加载 prefix 数据（如果提供）
        self.prefix_data = None
        if prefix_file and Path(prefix_file).exists():
            import json
            with open(prefix_file, "r") as f:
                self.prefix_data = json.load(f)
            print(f"Loaded {len(self.prefix_data)} prefix entries from {prefix_file}")
            # 创建 question -> prefix_data 的映射
            self.prefix_map = {item["question"]: item for item in self.prefix_data}
        elif prefix_file:
            print(f"Warning: Prefix file {prefix_file} not found, prefix mode will be disabled")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        q = item["question"]
        a = item["answer"]
        item.update(self.encode_prefix(q))
        
        # 如果启用了 prefix 模式，添加 prefix 数据
        if self.prefix_data is not None:
            if q in self.prefix_map:
                item["prefix_data"] = self.prefix_map[q]
            else:
                item["prefix_data"] = None  # 如果没有找到对应的 prefix，设为 None
        
        return item

    def encode_prefix(self, question: str):
        """
        构建原始的训练 prompt（base prompt）。
        
        注意：这个方法的名字可能容易误解。它构建的是：
        - 正常训练模式：完整的 prompt（要求模型输出完整推理和答案）
        - Prefix 训练模式：原始 prompt（后面会添加生成的 prefix）
        
        使用 USER_TEMPLATE 是合理的，因为：
        1. 正常训练模式需要完整答案
        2. Prefix 训练模式中，continuation 也需要完成完整答案
        """
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
            prefix_data=[x.get("prefix_data") for x in batch],  # 包含 prefix 数据（如果有）
        )

def formatpenalty(response: str, end_token: Optional[str] = None) -> float:
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
        return 0.0

    penalty = 0.0

    if think_match:
        penalty -= 0.1

    if answer_match:
        penalty -= 0.5

    return penalty


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
    where answer_reward is correctness of the GSM8K final answer
    """

    # keep your format reward unchanged
    format_penalty = formatpenalty("<think>" + response, end_token)

    # GSM8K correctness reward
    answer_reward = answer_reward_function_gsm8k(response, gold_answer=answer)

    return {
        "reward": format_penalty + answer_reward,
        "reward_info": {
            "format_reward": format_penalty,
            "answer_reward": answer_reward,
        },
    }


def generate_prefix_with_deepseek(
    question: str,
    tokenizer: Tokenizer,
    teacher_model: str = "deepseek/deepseek-r1-0528:free",
    api_key: Optional[str] = None,
) -> tuple[str, List[int], List[str]]:
    """
    使用 DeepSeek (OpenRouter) 生成 prefix。
    只按两句话截断，不限制 token 数量。
    
    Args:
        question: 问题文本
        tokenizer: Tokenizer 实例
        teacher_model: Teacher 模型名称（OpenRouter）
        api_key: OpenRouter API key（可选，从环境变量读取）
    
    Returns:
        (prefix_text, prefix_token_ids, prefix_tokens)
    """
    from openrouter_client import call_openrouter_simple
    
    user_message = PREFIX_USER_TEMPLATE.format(question=question)
    
    # 调试：打印调用参数
    # print(f"DEBUG: Calling DeepSeek API")
    # print(f"  Model: {teacher_model}")
    # print(f"  System prompt: {SYSTEM_MESSAGE[:50]}...")
    # print(f"  User message: {user_message[:100]}...")
    
    # 调用 OpenRouter API（设置较大的 max_tokens，确保能生成足够内容）
    # 注意：测试显示使用 system_prompt 时 API 能正常返回，不使用则返回空
    try:
        # 注意：确保所有参数都正确传递
        response = call_openrouter_simple(
            prompt=user_message,
            model=teacher_model,
            system_prompt=SYSTEM_MESSAGE,  # 使用 system_prompt（测试显示这是必需的）
            temperature=1.2,
            max_tokens=200,  # 设置较大的值，确保能生成足够内容
            api_key=api_key,
            timeout=120.0,  # 增加超时时间到 120 秒（某些模型可能需要更长时间）
            max_retries=3,  # 明确设置重试次数
        )
        
        # 调试：检查原始响应
        if not response or not response.strip():
            print(f"Warning: DeepSeek returned empty or None response.")
            print(f"  Model: {teacher_model}")
            print(f"  Has system_prompt: {SYSTEM_MESSAGE is not None}")
            print(f"  Raw response: {repr(response)}")
            print(f"  Response type: {type(response)}")
            return "", [], []
        
        # 清理响应文本（移除特殊 token 和乱码）
        cleaned_response = clean_prefix_text(response.strip())
        
        # 如果清理后为空，返回空 prefix
        if not cleaned_response:
            print(f"Warning: DeepSeek response became empty after cleaning. Original: {repr(response[:100])}")
            return "", [], []
        
        response = cleaned_response
        
        # 按句号截断，保留前2句（不限制 token 数量）
        prefix_text = truncate_by_sentences(response, max_sentences=2)
        
        # 只 tokenize 一次，获取 token_ids 和 tokens
        prefix_encoded = tokenizer.tokenize(prefix_text)
        prefix_token_ids = prefix_encoded.ids
        prefix_tokens = prefix_encoded.tokens
        
        return prefix_text, prefix_token_ids, prefix_tokens
    except Exception as e:
        # 如果 API 调用失败，返回空 prefix
        print(f"Warning: DeepSeek prefix generation failed: {e}")
        return "", [], []


def generate_prefix_with_3b(
    question: str,
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> tuple[str, List[int], List[str], List[float]]:
    """
    使用 3B 模型采样生成 prefix。
    只按两句话截断，不限制 token 数量。
    
    Args:
        question: 问题文本
        model: 3B 模型实例
        tokenizer: Tokenizer 实例
        device: 设备
        dtype: 数据类型
        temperature: 采样温度
    
    Returns:
        (prefix_text, prefix_token_ids, prefix_tokens, prefix_log_probs)
    """
    import torch
    
    # 构建 prompt
    user_message = PREFIX_USER_TEMPLATE.format(question=question)
    prefix_prompt = tokenizer.encode_chat_with_response_prompt(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ],
        RESPONSE_PROMPT,
    )
    prefix_prompt_tokens = tokenizer.tokenize(prefix_prompt)
    prefix_prompt_token_ids = prefix_prompt_tokens.ids
    
    # 准备输入
    prompt_tensor = torch.tensor([prefix_prompt_token_ids], dtype=torch.long, device=device)
    
    # 初始化 KV cache（设置较大的 max_seq_len，不限制生成长度）
    max_seq_len = len(prefix_prompt_token_ids) + 200  # 设置较大的上限
    model.init_kv_cache(max_batch_size=1, max_seq_len=max_seq_len, device=device, dtype=dtype)
    
    # 采样 prefix（检测句号，在第二句后停止）
    prefix_token_ids = []
    prefix_log_probs = []
    prev_pos = 0
    
    # 获取特殊 token IDs（用于过滤）
    # 尝试获取 endoftext token ID（如果存在）
    try:
        endoftext_token_id = tokenizer.tokenizer.token_to_id("<|endoftext|>")
    except:
        endoftext_token_id = None
    
    # 采样直到生成2句或达到上限
    for cur_pos in range(len(prefix_prompt_token_ids), max_seq_len):
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(prompt_tensor[:, prev_pos:cur_pos], prev_pos)
        
        # 采样下一个 token
        log_probs = torch.log_softmax(logits[:, -1] / temperature, dim=-1)
        
        # 过滤掉特殊 token（如果可能）
        if endoftext_token_id is not None:
            # 将特殊 token 的概率设为极小值
            log_probs[0, endoftext_token_id] = float('-inf')
        
        probs = torch.exp(log_probs)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # 如果采样到 EOS 或特殊 token，停止
        if next_token == tokenizer.eos_token_id:
            break
        if endoftext_token_id is not None and next_token == endoftext_token_id:
            break
        
        # 存储 log prob
        prefix_log_probs.append(log_probs[0, next_token].item())
        prefix_token_ids.append(next_token)
        
        # 检测句号：每生成几个 token 检查一次句号数量
        # 注意：这里需要 detokenize 来检测句号，但只在检测时使用，不是 tokenize → detokenize → tokenize
        should_check = (len(prefix_token_ids) % 5 == 0) or (len(prefix_token_ids) >= 10)
        if should_check:
            current_text = tokenizer.detokenize(prefix_token_ids)
            # 清理文本后再检测句号
            current_text = clean_prefix_text(current_text)
            # 修复句点检测：只有 . 后面有空格才能认为是句点
            sentence_count = len(re.findall(r'(?:[。!?]|\.\s+)(?:\s|$)', current_text))
            # 如果已经生成2句，停止采样
            if sentence_count >= 2:
                break
        
        # 更新 prompt_tensor（用于下一次推理）
        prompt_tensor = torch.cat([prompt_tensor, torch.tensor([[next_token]], device=device)], dim=1)
        prev_pos = cur_pos
    
    model.del_kv_cache()
    
    # 转换为文本并清理
    prefix_text = tokenizer.detokenize(prefix_token_ids)
    prefix_text = clean_prefix_text(prefix_text)  # 清理特殊 token 和乱码
    prefix_text = truncate_by_sentences(prefix_text, max_sentences=2)  # 按句号截断
    
    # 如果清理后文本为空，返回空 prefix
    if not prefix_text or not prefix_text.strip():
        return "", [], [], []
    
    # 只 tokenize 一次，获取截断后文本的 token_ids 和 tokens
    prefix_encoded = tokenizer.tokenize(prefix_text)
    prefix_token_ids = prefix_encoded.ids
    prefix_tokens = prefix_encoded.tokens
    
    # 调整 log_probs：需要找到截断点在原始 token_ids 中的位置
    # 由于截断可能改变 tokenization，我们需要重新计算 log_probs
    # 但为了保持一致性，我们只保留截断前的 log_probs（如果长度匹配）
    if len(prefix_log_probs) >= len(prefix_token_ids):
        prefix_log_probs = prefix_log_probs[:len(prefix_token_ids)]
    else:
        # 如果截断后 token 数量增加（理论上不应该），则填充 0
        prefix_log_probs = prefix_log_probs + [0.0] * (len(prefix_token_ids) - len(prefix_log_probs))
    
    return prefix_text, prefix_token_ids, prefix_tokens, prefix_log_probs