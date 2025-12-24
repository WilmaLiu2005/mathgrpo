"""
测试单个 prefix 生成调用（模拟 generate_prefixes.py 中的调用方式）
"""
import os
from pathlib import Path
import yaml
from countdown_task import generate_prefix_with_deepseek
from tokenizer import Tokenizer

def test_single_prefix():
    """测试单个 prefix 生成"""
    # 加载配置
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化 tokenizer
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    
    # 获取 API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        return
    
    teacher_model = config["training"].get("prefix_teacher_model", "deepseek/deepseek-r1-0528:free")
    
    # 测试问题
    test_question = "Janet has 8 more apples than John. John has 5 apples. How many apples does Janet have?"
    
    print("=" * 80)
    print("Testing single prefix generation (same as generate_prefixes.py)")
    print("=" * 80)
    print(f"Question: {test_question}")
    print(f"Model: {teacher_model}")
    print(f"API key: {api_key[:10]}... (truncated)")
    print("\nCalling generate_prefix_with_deepseek...")
    print("=" * 80)
    
    try:
        prefix_text, prefix_token_ids, prefix_tokens = generate_prefix_with_deepseek(
            question=test_question,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            api_key=api_key,
        )
        
        print(f"\nResult:")
        print(f"  Prefix text: {repr(prefix_text)}")
        print(f"  Prefix length: {len(prefix_text) if prefix_text else 0}")
        print(f"  Token IDs length: {len(prefix_token_ids)}")
        print(f"  Tokens length: {len(prefix_tokens)}")
        
        if prefix_text and prefix_text.strip():
            print(f"\n✓ Success! Prefix generated:")
            print(f"  {prefix_text}")
        else:
            print(f"\n✗ Failed! Empty prefix returned.")
            
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_prefix()

