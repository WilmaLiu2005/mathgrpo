#!/usr/bin/env python3
import json
from pathlib import Path

# 加载prefix文件
prefix_file = "/data/liuxinyu/prefixes.json"
with open(prefix_file, "r") as f:
    prefix_data = json.load(f)

# 创建question映射
prefix_map = {item["question"]: item for item in prefix_data}
print(f"Loaded {len(prefix_map)} prefix entries")

# 从eval结果中读取一个question
eval_file = "/home/student008/GRPO-zero/eval_results/20251226-103652_deepseek_prefix_1226/eval_results_step_000010.jsonl"
with open(eval_file, "r") as f:
    first_line = f.readline()
    eval_result = json.loads(first_line)
    test_question = eval_result["question"]

print(f"\nTest question from eval:")
print(f"{test_question[:100]}...")
print(f"\nQuestion in prefix_map: {test_question in prefix_map}")

# 检查是否有相似的question
if test_question not in prefix_map:
    print("\nSearching for similar questions...")
    # 检查是否有部分匹配
    for q in list(prefix_map.keys())[:10]:
        if test_question[:50] in q or q[:50] in test_question:
            print(f"Found similar: {q[:100]}...")
            break
else:
    prefix_info = prefix_map[test_question]
    print(f"\nPrefix info found:")
    print(f"  deepseek_prefixes: {len(prefix_info.get('deepseek_prefixes', []))}")
    print(f"  3b_prefixes: {len(prefix_info.get('3b_prefixes', []))}")

