from openrouter_client import call_openrouter_simple

# 简单调用
response = call_openrouter_simple(
    "What is 2+2?",
    api_key='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',
    model="deepseek/deepseek-r1-0528:free",
    temperature=1.2
)
print(response)