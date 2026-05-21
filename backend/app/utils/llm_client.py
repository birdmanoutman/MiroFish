"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import fcntl
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from openai import DefaultHttpxClient, OpenAI

from ..config import Config


def _message_text(message: Any) -> str:
    values = [getattr(message, "content", None), getattr(message, "reasoning_content", None)]
    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
            values.extend([dumped.get("content"), dumped.get("reasoning_content")])
        except Exception:
            pass
    for value in values:
        if value:
            return str(value).strip()
    return ""


def _wait_for_glm_rate_slot(model: str) -> None:
    if "glm" not in str(model).lower():
        return

    limit = max(1, Config.GRAPHITI_LLM_RPM_LIMIT)
    window_seconds = max(1.0, Config.GRAPHITI_LLM_RATE_WINDOW_SECONDS)
    min_interval_seconds = max(0.0, Config.GRAPHITI_LLM_MIN_INTERVAL_SECONDS)
    rate_path = Path(Config.GRAPHITI_LLM_RATE_LIMIT_PATH)
    rate_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = rate_path.with_suffix(rate_path.suffix + ".lock")

    while True:
        now = time.monotonic()
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    payload = json.loads(rate_path.read_text(encoding="utf-8"))
                    timestamps = [float(item) for item in payload.get("timestamps", [])]
                except Exception:
                    timestamps = []

                timestamps = [ts for ts in timestamps if now - ts < window_seconds]
                wait_for_interval = 0.0
                if timestamps and min_interval_seconds:
                    wait_for_interval = min_interval_seconds - (now - max(timestamps))
                wait_for_window = 0.0
                if len(timestamps) >= limit:
                    wait_for_window = window_seconds - (now - min(timestamps))

                wait_seconds = max(wait_for_interval, wait_for_window, 0.0)
                if wait_seconds <= 0:
                    timestamps.append(now)
                    rate_path.write_text(
                        json.dumps({"timestamps": timestamps}, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    return
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        time.sleep(max(0.1, wait_seconds))


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.timeout = float(os.environ.get("MIROFISH_LLM_TIMEOUT_SECONDS", "180"))
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=DefaultHttpxClient(timeout=self.timeout, trust_env=False),
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format

        try:
            _wait_for_glm_rate_slot(self.model)
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not response_format or ("response_format.type" not in str(exc) and "json_object" not in str(exc)):
                raise
            kwargs.pop("response_format", None)
            _wait_for_glm_rate_slot(self.model)
            response = self.client.chat.completions.create(**kwargs)
        content = _message_text(response.choices[0].message)
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            start = cleaned_response.find("{")
            end = cleaned_response.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned_response[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
