"""
Prefix 生成脚本：为每个问题生成多个 prefix（DeepSeek 和 3B），供训练时使用。
"""
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from countdown_task import (
    SYSTEM_MESSAGE,
    PREFIX_USER_TEMPLATE,
    clean_prefix_text,
    generate_prefix_with_3b,
    generate_prefix_with_deepseek,
    truncate_by_sentences,
)
from qwen2_model import Transformer
from tokenizer import Tokenizer


def _generate_single_deepseek_prefix(
    question: str,
    tokenizer: Tokenizer,
    teacher_model: str,
    api_key: Optional[str],
    idx: int,
    prefix_idx: int,
) -> Tuple[int, int, Optional[Tuple[str, List[int], List[str]]]]:
    """
    生成单个 DeepSeek prefix（用于并发调用）。
    
    Returns:
        (question_idx, prefix_idx, result) 或 (question_idx, prefix_idx, None) 如果失败
    """
    try:
        # 检查 API key
        if not api_key:
            print(f"Error: API key is None for question {idx}, prefix {prefix_idx+1}")
            return (idx, prefix_idx, None)
        
        prefix_text, prefix_token_ids, prefix_tokens = generate_prefix_with_deepseek(
            question=question,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            api_key=api_key,
        )
        if prefix_text and prefix_text.strip():
            return (idx, prefix_idx, (prefix_text, prefix_token_ids, prefix_tokens))
        else:
            # 添加更详细的调试信息（只在第一个失败时打印，避免输出过多）
            if prefix_idx == 0:  # 只打印第一个 prefix 的失败信息
                print(f"Warning: Empty prefix for question {idx}, prefix {prefix_idx+1}")
                print(f"  Question: {question[:50]}...")
                print(f"  Prefix text: {repr(prefix_text)}")
                print(f"  Prefix length: {len(prefix_text) if prefix_text else 0}")
            return (idx, prefix_idx, None)
    except Exception as e:
        import traceback
        # 只在第一个失败时打印完整错误，避免输出过多
        if prefix_idx == 0:
            print(f"Warning: Failed to generate DeepSeek prefix {prefix_idx+1} for question {idx}: {e}")
            print(f"  Question: {question[:50]}...")
            traceback.print_exc()
        return (idx, prefix_idx, None)


def generate_prefixes_for_dataset(
    config_path: str,
    output_path: str,
    num_deepseek: int = 5,
    num_3b: int = 5,
    max_workers: int = 10,  # DeepSeek API 并发数
):
    """
    为数据集中的每个问题生成 prefix。
    
    Args:
        config_path: 配置文件路径
        output_path: 输出文件路径（JSON格式）
        num_deepseek: 每个问题生成多少个 DeepSeek prefix
        num_3b: 每个问题生成多少个 3B prefix
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化模型和 tokenizer
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    
    # 加载 3B 模型（用于生成 3B prefix）
    model = Transformer.from_pretrained(pretrained_model_path, device=device).eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # 加载数据集
    train_dataset = pd.read_parquet(
        Path(config["data"]["path"]) / config["data"].get("config_name", "main") / "train.parquet"
    )
    
    teacher_model = config["training"].get("prefix_teacher_model", "deepseek/deepseek-r1-0528:free")
    api_key = os.getenv("OPENROUTER_API_KEY")
    temperature = config["training"].get("temperature", 1.0)
    
    # 检查 API key
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it before running.")
    print(f"Using API key: {api_key[:10]}... (truncated for security)")
    
    # 存储结果
    prefix_data = []
    
    print(f"Generating prefixes for {len(train_dataset)} questions...")
    print(f"Each question: {num_deepseek} DeepSeek prefixes + {num_3b} 3B prefixes")
    print(f"Using {max_workers} concurrent workers for DeepSeek API calls")
    
    # 初始化所有问题的 prefix 字典
    all_question_prefixes = {}
    for idx, row in train_dataset.iterrows():
        all_question_prefixes[idx] = {
            "question": row["question"],
            "answer": row["answer"],
            "deepseek_prefixes": [None] * num_deepseek,  # 预分配位置
            "3b_prefixes": [],
        }
    
    # 并发生成所有 DeepSeek prefixes
    print("Generating DeepSeek prefixes (concurrent)...")
    deepseek_tasks = []
    for idx, row in train_dataset.iterrows():
        question = row["question"]
        for i in range(num_deepseek):
            deepseek_tasks.append((idx, question, i))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_single_deepseek_prefix,
                question,
                tokenizer,
                teacher_model,
                api_key,
                idx,
                i,
            ): (idx, i)
            for idx, question, i in deepseek_tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek prefixes"):
            idx, prefix_idx = futures[future]
            result = future.result()
            if result[2] is not None:  # 成功生成
                prefix_text, prefix_token_ids, prefix_tokens = result[2]
                all_question_prefixes[idx]["deepseek_prefixes"][prefix_idx] = {
                    "text": prefix_text,
                    "token_ids": prefix_token_ids,
                    "tokens": prefix_tokens,
                }
            # 如果失败，保持为 None，后续会过滤掉
    
    # 过滤掉失败的 DeepSeek prefixes（None 值）
    for idx in all_question_prefixes:
        all_question_prefixes[idx]["deepseek_prefixes"] = [
            p for p in all_question_prefixes[idx]["deepseek_prefixes"] if p is not None
        ]
    
    # 串行生成 3B prefixes（因为需要 GPU，并发可能受限于显存）
    print("Generating 3B prefixes (sequential)...")
    for idx, row in tqdm(train_dataset.iterrows(), total=len(train_dataset), desc="3B prefixes"):
        question = row["question"]
        
        # 生成 3B prefixes
        for i in range(num_3b):
            try:
                prefix_text, prefix_token_ids, prefix_tokens, prefix_log_probs = generate_prefix_with_3b(
                    question=question,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dtype=dtype,
                    temperature=temperature,
                )
                if prefix_text and prefix_text.strip():
                    all_question_prefixes[idx]["3b_prefixes"].append({
                        "text": prefix_text,
                        "token_ids": prefix_token_ids,
                        "tokens": prefix_tokens,
                        "log_probs": prefix_log_probs,
                    })
            except Exception as e:
                print(f"Warning: Failed to generate 3B prefix {i+1} for question {idx}: {e}")
                continue
        
        # 每100个问题保存一次（防止中途中断丢失数据）
        if (idx + 1) % 100 == 0:
            prefix_data = [all_question_prefixes[i] for i in sorted(all_question_prefixes.keys())]
            with open(output_path, "w") as f:
                json.dump(prefix_data, f, indent=2, ensure_ascii=False)
            print(f"Saved progress: {idx + 1}/{len(train_dataset)} questions")
    
    # 最终保存
    prefix_data = [all_question_prefixes[i] for i in sorted(all_question_prefixes.keys())]
    with open(output_path, "w") as f:
        json.dump(prefix_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nPrefix generation completed! Saved to {output_path}")
    print(f"Total questions: {len(prefix_data)}")
    print(f"Average DeepSeek prefixes per question: {sum(len(p['deepseek_prefixes']) for p in prefix_data) / len(prefix_data):.2f}")
    print(f"Average 3B prefixes per question: {sum(len(p['3b_prefixes']) for p in prefix_data) / len(prefix_data):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default="prefixes.json", help="Output JSON file path")
    parser.add_argument("--num-deepseek", type=int, default=5, help="Number of DeepSeek prefixes per question")
    parser.add_argument("--num-3b", type=int, default=5, help="Number of 3B prefixes per question")
    parser.add_argument("--max-workers", type=int, default=10, help="Max concurrent workers for DeepSeek API calls")
    
    args = parser.parse_args()
    
    generate_prefixes_for_dataset(
        config_path=args.config,
        output_path=args.output,
        num_deepseek=args.num_deepseek,
        num_3b=args.num_3b,
        max_workers=args.max_workers,
    )

