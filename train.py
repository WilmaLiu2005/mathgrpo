import html
import time
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from countdown_task import GSM8KDataset, reward_function
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer
import wandb
import random
from copy import deepcopy
import os

def sample_trace_by_length_group(episodes):
    groups = {"short": [], "medium": [], "long": []}
    
    # Compute length distribution statistics
    lengths = [len(ep.generated_token_ids) for ep in episodes]
    if len(lengths) > 0:
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        # Handle edge case where all lengths are the same (std = 0)
        if std_length < 1e-6:
            std_length = 1.0  # Use a small default std to avoid division issues
        # Group by: < mean - sigma (short), mean - sigma to mean + sigma (medium), > mean + sigma (long)
        lower_bound = mean_length - std_length
        upper_bound = mean_length + std_length
    else:
        # Fallback if no episodes
        lower_bound = 0
        upper_bound = float('inf')
    
    for ep in episodes:
        length = len(ep.generated_token_ids)
        if length < lower_bound:
            groups["short"].append(ep)
        elif length <= upper_bound:
            groups["medium"].append(ep)
        else:
            groups["long"].append(ep)
    sampled = {}
    for k, v in groups.items():
        if v:
            sampled[k] = random.choice(v)
    return sampled

def evaluate(model, tokenizer, device, dtype, config, global_step, timestamp=None, safe_run_name=None):
    test_dataset = GSM8KDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        config_name=config["data"].get("config_name", "main"),
        test_size=config["data"]["test_size"],
        prefix_file=config["training"].get("prefix_file", None),  # Prefix 文件路径（如果启用）
    )
    generator = torch.Generator(device=device)
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=GSM8KDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    lengths = []
    all_episodes = []
    all_eval_results = []  # 用于保存评估结果
    
    for batch in dataloader:
            episodes = rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=config["training"]["max_gen_len"] * 2,
                num_answer_per_question=1,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
                temperature=config["training"].get("temperature", 1.0),
                enable_prefix=config["training"].get("enable_prefix", False),
                prefix_dropout_prob=config["training"].get("prefix_dropout_prob", 0.5),
                prefix_use_prob=0.0,  # eval时总是设为0.0，不使用prefix（与训练分布一致）
            )
            all_episodes.extend(episodes)
            success.extend([episode.reward_info["answer_reward"] for episode in episodes])
            lengths.extend([len(episode.generated_token_ids) for episode in episodes])
            
            # 收集评估结果详细信息
            # 注意：由于 num_answer_per_question=1，episodes 的长度应该等于 batch.questions 的长度
            assert len(episodes) == len(batch.questions), f"Episodes length ({len(episodes)}) != batch questions length ({len(batch.questions)})"
            for i, episode in enumerate(episodes):
                eval_result = {
                    "global_step": global_step,
                    "question": batch.questions[i],
                    "gold_answer": batch.answers[i],
                    "generated_text": episode.text,
                    "prefix": episode.prefix,
                    "is_finished": episode.is_finished,
                    "reward": float(episode.reward),
                    "format_reward": float(episode.reward_info.get("format_reward", 0.0)),
                    "answer_reward": float(episode.reward_info.get("answer_reward", 0.0)),
                    "is_correct": bool(episode.reward_info.get("answer_reward", 0.0) > 0.0),
                    "response_length": len(episode.generated_token_ids),
                    "prefix_length": episode.prefix_length,
                    "prefix_source": episode.prefix_source,
                }
                all_eval_results.append(eval_result)
        
    # 采样trace
    sampled_traces = sample_trace_by_length_group(all_episodes)
    for group, ep in sampled_traces.items():
        print(f"[{group}] trace: {ep.text[:200]} ...")  # 只打印前200字符
        wandb.log({f"eval_trace/{group}": wandb.Html(f"<pre>{ep.text}</pre>")}, step=global_step)
    
    # 保存评估结果到 JSONL 文件
    # 使用传入的时间戳和 run_name（如果未提供则生成新的）
    if timestamp is None:
        timestamp = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    if safe_run_name is None:
        run_name = config["wandb"].get("run_name", "baseline")
        safe_run_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in run_name)
    
    base_dir = Path(config["training"].get("eval_results_dir", "eval_results"))
    eval_results_dir = base_dir / f"{timestamp}_{safe_run_name}"
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_results_dir / f"eval_results_step_{global_step:06d}.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in all_eval_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(all_eval_results)} evaluation results to {output_file}")
    
    return np.mean(success), np.mean(lengths)

def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---------------------------
    # Init wandb
    # ---------------------------
    wandb.init(
        project=config["wandb"].get("project", "mathgrpo"),
        name=config["wandb"].get("run_name", "baseline"),
        mode=config["wandb"].get("mode", "online"),
        config=config
    )

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])

    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    
    # 生成用于评估结果保存的时间戳和 run_name
    run_name = config["wandb"].get("run_name", "baseline")
    safe_run_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in run_name)

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = GSM8KDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        config_name=config["data"].get("config_name", "main"),
        test_size=config["data"]["test_size"],
        prefix_file=config["training"].get("prefix_file", None),  # Prefix 文件路径（如果启用）
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=GSM8KDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    use_similarity_weighting = config["training"].get("use_similarity_weighting", False)
    use_ref_for_similarity_weighting = config["training"].get("use_ref_for_similarity_weighting", False)

    ref_model = None
    if config["training"].get("use_kl_penalty", False) or use_ref_for_similarity_weighting:
        ref_model = deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False    

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 新增：多 epoch / 全局步数 控制
    # ---------------------------
    num_epochs = config["training"].get("num_epochs", 1)
    max_steps = config["training"].get("max_steps", None)
    if isinstance(max_steps, int) and max_steps <= 0:
        max_steps = None

    global_step = 0
    start_time = time.time()

    # For dynamic difficulty grouping
    question_accuracies = {}
    use_difficulty_sorting = config["training"].get("use_difficulty_sorting", False)

    for epoch in range(1, num_epochs + 1):
        for step_in_epoch, batch in enumerate(train_dataloader, start=1):
            global_step += 1

            episodes = rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=config["training"]["max_gen_len"],
                num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                reward_function=reward_function,
                device=device,
                dtype=dtype,
                temperature=config["training"].get("temperature", 1.0),
                enable_prefix=config["training"].get("enable_prefix", False),
                prefix_dropout_prob=config["training"].get("prefix_dropout_prob", 0.5),
                prefix_use_prob=config["training"].get("prefix_use_prob", 1.0),  # 训练时使用prefix的概率
            )

            if config["training"]["skip_unfinished_episodes"]:
                episodes = [ep for ep in episodes if ep.is_finished]

            # Collect accuracies for dynamic difficulty grouping during epoch 1
            if epoch == 1 and use_difficulty_sorting:
                for i, idx in enumerate(batch.indices):
                    start_idx = i * NUM_ANSWERS_PER_QUESTION
                    end_idx = start_idx + NUM_ANSWERS_PER_QUESTION
                    question_episodes = episodes[start_idx:end_idx]
                    if question_episodes:
                        avg_acc = np.mean([ep.reward_info.get("answer_reward", 0.0) for ep in question_episodes])
                        question_accuracies[idx] = float(avg_acc)

            process_rewards_kwargs = dict(device=device, dtype=dtype, pad_token_id=tokenizer.pad_token_id)
            process_rewards_kwargs["use_similarity_weighting"] = use_similarity_weighting
            if use_similarity_weighting:
                process_rewards_kwargs["model"] = ref_model if use_ref_for_similarity_weighting else model
                process_rewards_kwargs["fix_model"] = use_ref_for_similarity_weighting
                process_rewards_kwargs.update(config["training"].get("similarity_weighting_config", dict()))

            results = update_policy(
                model=model,
                optimizer=optimizer,
                episodes=episodes,
                micro_batch_size=config["training"]["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=config["training"]["max_grad_norm"],
                device=device,
                dtype=dtype,
                ref_model=ref_model if config["training"].get("use_kl_penalty", False) else None,
                epsilon_low=config["training"].get("epsilon_low", 0.2),
                epsilon_high=config["training"].get("epsilon_high", 0.2),
                kl_coeff=config["training"].get("kl_coeff", 0.0),
                use_length_grouping=config["training"].get("use_length_grouping", False),
                use_dynamic_clipping=config["training"].get("use_dynamic_clipping", True),
                use_kl_penalty=config["training"].get("use_kl_penalty", False),
                clip_ratio=config["training"].get("clip_ratio", 0.2),
                enable_prefix=config["training"].get("enable_prefix", False),
                prefix_sft_coeff=config["training"].get("prefix_sft_coeff", 0.2),
                process_reward_kwargs=process_rewards_kwargs,
            )

            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            # 统计与打印（用 global_step）
            reward = [ep.reward for ep in episodes]
            answer_reward = [ep.reward_info["answer_reward"] for ep in episodes]
            format_reward_list = [ep.reward_info["format_reward"] for ep in episodes]
            num_finished_episodes = sum(ep.is_finished for ep in episodes)

            mean_reward = float(np.mean(reward)) if reward else 0.0
            std_reward = float(np.std(reward)) if reward else 0.0
            success_rate = float(np.mean(answer_reward)) if answer_reward else 0.0
            mean_format_reward = float(np.mean(format_reward_list)) if format_reward_list else 0.0
            grad_norm = results["grad_norm"]
            entropy = results["entropy"]
            lr = optimizer.param_groups[0]["lr"]
            loss = results["loss"]
            kl_loss = results.get("kl_loss", 0.0)
            prefix_sft_loss = results.get("prefix_sft_loss", 0.0)
            mean_response_len = float(np.mean([len(ep.generated_token_ids) for ep in episodes])) if episodes else 0.0
            mean_prefix_len = float(np.mean([ep.prefix_length for ep in episodes])) if episodes and config["training"].get("enable_prefix", False) else 0.0
            
            # Clip statistics
            clip_frac_lower = results.get("clip_frac_lower", 0.0)
            clip_frac_upper = results.get("clip_frac_upper", 0.0)
            clip_frac_total = results.get("clip_frac_total", 0.0)
            mean_ratio = results.get("mean_ratio", 0.0)

            print(
                f"\rEpoch {epoch}, step {global_step}, "
                f"mean_reward: {mean_reward:.2f}, "
                f"train success_rate: {success_rate:.2f}, "
                f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
                f"num_finished_episodes: {num_finished_episodes}, "
                f"mean_response_len: {mean_response_len:.2f}, "
                f"entropy: {entropy:.2f}"
            )

            # Eval 按 global_step 触发
            if global_step % config["training"]["eval_interval"] == 0:
                eval_success_rate, eval_mean_len = evaluate(model, tokenizer, device, dtype, config, global_step, current_time, safe_run_name)
                print(f"\rEval success rate: {eval_success_rate:.2f}, Eval mean len: {eval_mean_len:.2f}" + " " * 100)
                tb_writer.add_scalar("success_rate/eval", eval_success_rate, global_step)
                tb_writer.add_scalar("mean_response_len/eval", eval_mean_len, global_step)
                wandb.log({"success_rate/eval": float(eval_success_rate), "mean_response_len/eval": float(eval_mean_len)}, step=global_step)

            # TensorBoard（用 global_step）
            tb_writer.add_scalar("loss", loss, global_step)
            if kl_loss > 0:
                tb_writer.add_scalar("kl_loss", kl_loss, global_step)
            if prefix_sft_loss > 0:
                tb_writer.add_scalar("prefix_sft_loss", prefix_sft_loss, global_step)
            if mean_prefix_len > 0:
                tb_writer.add_scalar("mean_prefix_len", mean_prefix_len, global_step)
            tb_writer.add_scalar("mean_reward", mean_reward, global_step)
            tb_writer.add_scalar("std_reward", std_reward, global_step)
            tb_writer.add_scalar("success_rate/train", success_rate, global_step)
            tb_writer.add_scalar("format_reward", mean_format_reward, global_step)
            tb_writer.add_scalar("grad_norm", grad_norm, global_step)
            tb_writer.add_scalar("duration", duration, global_step)
            tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, global_step)
            tb_writer.add_scalar("learning_rate", lr, global_step)
            tb_writer.add_scalar("mean_response_len", mean_response_len, global_step)
            tb_writer.add_scalar("entropy", entropy, global_step)
            # Clip statistics
            tb_writer.add_scalar("clip_frac_lower", clip_frac_lower, global_step)
            tb_writer.add_scalar("clip_frac_upper", clip_frac_upper, global_step)
            tb_writer.add_scalar("clip_frac_total", clip_frac_total, global_step)
            tb_writer.add_scalar("mean_ratio", mean_ratio, global_step)

            # Wandb（用 global_step）
            log_dict = {
                "loss": loss,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "success_rate/train": success_rate,
                "format_reward": mean_format_reward,
                "grad_norm": grad_norm,
                "duration": duration,
                "num_finished_episodes": num_finished_episodes,
                "learning_rate": lr,
                "mean_response_len": mean_response_len,
                "entropy": entropy,
                "clip_frac_lower": clip_frac_lower,
                "clip_frac_upper": clip_frac_upper,
                "clip_frac_total": clip_frac_total,
                "mean_ratio": mean_ratio,
            }
            if kl_loss > 0:
                log_dict["kl_loss"] = kl_loss
            if prefix_sft_loss > 0:
                log_dict["prefix_sft_loss"] = prefix_sft_loss
            if mean_prefix_len > 0:
                log_dict["mean_prefix_len"] = mean_prefix_len
            wandb.log(log_dict, step=global_step)

            # 文本样例（用 global_step）
            for i, ep in enumerate(episodes):
                text = html.escape(ep.text)
                tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", global_step)

            # 保存 checkpoint 按 global_step 触发
            if global_step % config["training"]["ckpt_save_interval"] == 0:
                output_file = ckpt_dir / f"ckpt_{global_step:06d}.pt"
                torch.save(model.state_dict(), output_file)
                print(f"Saved checkpoint to {output_file}")
                # 可选：把 ckpt 上传到 wandb
                # wandb.save(str(output_file))

            # 如设置了 max_steps，则达到后提前结束
            if max_steps is not None and global_step >= max_steps:
                break

        # End of epoch logic
        if epoch == 1 and use_difficulty_sorting:
            print(f"\nEpoch 1 finished. Sorting dataset by difficulty for subsequent epochs...")
            # Update dataset with collected accuracies
            accuracies = [question_accuracies.get(i, 0.0) for i in range(len(train_dataset))]
            train_dataset.data = train_dataset.data.copy()
            train_dataset.data['accuracy'] = accuracies
            # Sort by accuracy descending (simple first)
            train_dataset.data = train_dataset.data.sort_values(by='accuracy', ascending=False)
            
            print(f"Top 5 accuracies: {train_dataset.data['accuracy'].head().tolist()}")
            print(f"Bottom 5 accuracies: {train_dataset.data['accuracy'].tail().tolist()}")
            
            # Re-create train_dataloader with shuffle=False for curriculum
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=False,
                collate_fn=GSM8KDataset.collate_fn,
                generator=generator,
                batch_size=NUM_QUESTIONS_PER_BATCH,
            )
            print("Dataset sorted and dataloader updated.")

        if max_steps is not None and global_step >= max_steps:
            break

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
