from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")  # можно заменить на другую NIM модель
NVIDIA_API_URL = f"https://api.nvidia.com/v1/models/{NVIDIA_MODEL}/completions"

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    data = await request.json()

    # Преобразуем OpenAI-совместимый формат Janitor в формат NIM
    messages = data.get("messages", [])
    user_input = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    payload = {
        "model": NVIDIA_MODEL,
        "prompt": user_input,
        "max_tokens": data.get("max_tokens", 512),
        "temperature": data.get("temperature", 0.7),
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(NVIDIA_API_URL, json=payload, headers=headers)

    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content={"error": response.text},
        )

    nvidia_data = response.json()
    text_output = nvidia_data.get("choices", [{}])[0].get("text", "").strip()

    # Преобразуем обратно в формат OpenAI для Janitor.ai
    return {
        "id": "chatcmpl-nvidia",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text_output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(text_output.split()),
            "total_tokens": len(text_output.split()),
        },
    }

@app.get("/")
async def root():
    return {"message": "Nvidia NIM proxy running!"}
