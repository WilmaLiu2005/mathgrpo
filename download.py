#!/usr/bin/env python3
"""
Download GSM8K, Countdown-Tasks-3to4 datasets and the full Qwen2.5-3B-Instruct model.
All files are downloaded into their respective root directories, keeping subfolder structure.
"""

from datasets import load_dataset
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

BASE_DIR = Path("/home/student008/GRPO-Zero")

# 根目录
GSM8K_DIR = BASE_DIR / "gsm8k"
COUNTDOWN_DIR = BASE_DIR / "Countdown-Tasks-3to4"
QWEN_DIR = BASE_DIR / "Qwen2.5-3B-Instruct"

# ---------- GSM8K 下载 ---------- #
def download_gsm8k():
    print("\nDownloading GSM8K main...")
    ds_main = load_dataset("openai/gsm8k", "main")
    main_dir = GSM8K_DIR / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    ds_main["train"].to_parquet(main_dir / "train.parquet")
    ds_main["test"].to_parquet(main_dir / "test.parquet")
    print("Main done.")

    print("\nDownloading GSM8K socratic...")
    ds_socratic = load_dataset("openai/gsm8k", "socratic")
    socratic_dir = GSM8K_DIR / "socratic"
    socratic_dir.mkdir(parents=True, exist_ok=True)
    ds_socratic["train"].to_parquet(socratic_dir / "train.parquet")
    ds_socratic["test"].to_parquet(socratic_dir / "test.parquet")
    print("Socratic done.")

# ---------- Countdown 下载 ---------- #
def download_countdown():
    print("\nDownloading Countdown-Tasks-3to4 ...")
    ds_countdown = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
    COUNTDOWN_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = COUNTDOWN_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if "train" in ds_countdown:
        ds_countdown["train"].to_parquet(data_dir / "train.parquet")
    if "test" in ds_countdown:
        ds_countdown["test"].to_parquet(data_dir / "test.parquet")
    print("Countdown download finished.")

# ---------- Qwen 全量下载 ---------- #
def download_qwen_full():
    print("\nDownloading full Qwen2.5-3B-Instruct model ...")
    QWEN_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取 repo 中所有文件
    files = list_repo_files("Qwen/Qwen2.5-3B-Instruct", repo_type="model")
    for f in files:
        local_path = QWEN_DIR / f
        if local_path.exists():
            continue  # 已经存在
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {f} ...")
        try:
            hf_hub_download(
                repo_id="Qwen/Qwen2.5-3B-Instruct",
                filename=f,
                cache_dir=str(QWEN_DIR),
                force_download=False,
                resume_download=True,
                repo_type="model",
            )
            print(f"Downloaded {f}")
        except Exception as e:
            print(f"Failed to download {f}: {e}")
            print("Please download manually if needed.")
    print("Qwen download finished.")

# ---------- 主函数 ---------- #
if __name__ == "__main__":
    download_gsm8k()
    download_countdown()
    download_qwen_full()
    print("\nAll done.")
