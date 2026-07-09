# utils_llm.py
import time
from openai import OpenAI
import config

def get_openai_client():
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL
    )

def query_astron(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_retries: int = 3) -> str:
    """
    通用大模型请求封装（通过将 max_tokens 放大至 2048 并利用 json_object 模式，从源头减少 JSON 截断风险）
    """
    client = get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2048,  # 👈 从 1024 放大到 2048，确保完整的 Reason 生成空间
                response_format={"type": "json_object"}  # 👈 强制模型返回标准 JSON
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f" ⚠️ 请求 Astron 模型出错 (第 {attempt + 1}/{max_retries} 次尝试): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.2)
            else:
                print(" ❌ 已达到最大重试次数，返回空响应。")
                return ""