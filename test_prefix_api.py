"""
测试 DeepSeek API 是否能正常响应 prefix 生成的 prompt
"""
import os
from openrouter_client import call_openrouter_simple

# 使用与 countdown_task.py 中相同的 prompt 模板
SYSTEM_MESSAGE = (
    "You are a helpful assistant that solves math problems accurately."
)

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

def test_prefix_api():
    """测试 prefix API 调用"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        return
    
    # 使用一个示例问题
    test_question = "Janet has 8 more apples than John. John has 5 apples. How many apples does Janet have?"
    
    # 构建 user message（与 generate_prefix_with_deepseek 中相同）
    user_message = PREFIX_USER_TEMPLATE.format(question=test_question)
    
    print("=" * 80)
    print("Testing DeepSeek API with prefix generation prompt")
    print("=" * 80)
    print(f"\nModel: deepseek/deepseek-r1-0528:free")
    print(f"\nSystem prompt: {SYSTEM_MESSAGE}")
    print(f"\nUser message:")
    print(user_message)
    print("\n" + "=" * 80)
    print("Calling API...")
    print("=" * 80)
    
    try:
        # 测试 1: 不使用 system_prompt（与当前代码一致）
        print("\n[Test 1] Without system_prompt:")
        response1 = call_openrouter_simple(
            prompt=user_message,
            model="deepseek/deepseek-r1-0528:free",
            system_prompt=None,  # 与当前代码一致
            temperature=1.2,
            max_tokens=200,
            api_key=api_key,
            timeout=60.0,
            max_retries=3,
        )
        print(f"Response: {repr(response1)}")
        print(f"Response length: {len(response1) if response1 else 0}")
        print(f"Response (first 200 chars): {response1[:200] if response1 else 'None'}")
        
        # 测试 2: 使用 system_prompt（对比测试）
        print("\n[Test 2] With system_prompt:")
        response2 = call_openrouter_simple(
            prompt=user_message,
            model="deepseek/deepseek-r1-0528:free",
            system_prompt=SYSTEM_MESSAGE,
            temperature=1.2,
            max_tokens=200,
            api_key=api_key,
            timeout=60.0,
            max_retries=3,
        )
        print(f"Response: {repr(response2)}")
        print(f"Response length: {len(response2) if response2 else 0}")
        print(f"Response (first 200 chars): {response2[:200] if response2 else 'None'}")
        
        # 测试 3: 最简单的调用（与 test.py 一致）
        print("\n[Test 3] Simple call (like test.py):")
        response3 = call_openrouter_simple(
            "Hello",
            model="deepseek/deepseek-r1-0528:free"
        )
        print(f"Response: {repr(response3)}")
        print(f"Response length: {len(response3) if response3 else 0}")
        
        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prefix_api()

