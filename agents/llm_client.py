"""LLM client for Tiny Aya models — Cohere V2 API compatible."""

import json
import os
import time
from typing import Any, Optional, List, Dict

import httpx
from dotenv import load_dotenv

load_dotenv()

# Tiny Aya Configuration
TINY_AYA_API_KEY = os.getenv("TINY_AYA_API_KEY", "")
TINY_AYA_BASE_URL = os.getenv("TINY_AYA_BASE_URL", "https://api.cohere.com/v2")
TINY_AYA_MODEL = os.getenv("TINY_AYA_MODEL", "tiny-aya-global")
TINY_AYA_TEMPERATURE = float(os.getenv("TINY_AYA_TEMPERATURE", "0.2"))
TINY_AYA_MAX_TOKENS = int(os.getenv("TINY_AYA_MAX_TOKENS", "2048"))

# Available Tiny Aya Model Variants
TINY_AYA_MODELS = {
    "tiny-aya-global": "Tiny Aya Global — General purpose",
    "tiny-aya-fire": "Tiny Aya Fire — High creativity",
    "tiny-aya-earth": "Tiny Aya Earth — Grounded reasoning",
    "tiny-aya-water": "Tiny Aya Water — Fluid adaptability",
}

# Cohere V2 API endpoints
COHERE_CHAT_ENDPOINT = "/chat"
COHERE_MODELS_ENDPOINT = "/models"


class TinyAyaClient:
    """Cohere V2 API compatible client for Tiny Aya models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or TINY_AYA_API_KEY
        self.base_url = (base_url or TINY_AYA_BASE_URL).rstrip("/")
        self.model = model or TINY_AYA_MODEL
        self.temperature = temperature if temperature is not None else TINY_AYA_TEMPERATURE
        self.max_tokens = max_tokens or TINY_AYA_MAX_TOKENS
        self.timeout = timeout
        self._validate_config()
        self._client = None

    def _validate_config(self):
        """Validate the client configuration."""
        if not self.api_key:
            raise ValueError("TINY_AYA_API_KEY is required but not configured")
        if self.model not in TINY_AYA_MODELS:
            raise ValueError(
                f"Model '{self.model}' not supported. Available models: {list(TINY_AYA_MODELS.keys())}"
            )

    def get_available_models(self) -> Dict[str, str]:
        """Return available model variants."""
        return TINY_AYA_MODELS.copy()

    def set_model(self, model: str):
        """Change the active model."""
        if model not in TINY_AYA_MODELS:
            raise ValueError(
                f"Model '{model}' not supported. Available models: {list(TINY_AYA_MODELS.keys())}"
            )
        self.model = model

    def _get_client(self):
        """Get or create the Cohere V2 client."""
        if self._client is None:
            import cohere
            self._client = cohere.ClientV2(api_key=self.api_key)
        return self._client

    def get_available_models(self) -> Dict[str, str]:
        """Return available model variants."""
        return TINY_AYA_MODELS.copy()

    def set_model(self, model: str):
        """Change the active model."""
        if model not in TINY_AYA_MODELS:
            raise ValueError(
                f"Model '{model}' not supported. Available models: {list(TINY_AYA_MODELS.keys())}"
            )
        self.model = model

    def validate_connection(self) -> Dict[str, Any]:
        """Validate API connection and return status."""
        try:
            client = self._get_client()
            # Try to list models to validate connection
            resp = client.models.list()
            return {
                "connected": True,
                "model": self.model,
                "base_url": self.base_url,
                "available_models": list(TINY_AYA_MODELS.keys()),
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "model": self.model,
                "base_url": self.base_url,
            }

    def messages_create(
        self,
        model: str,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> "TinyAyaResponse":
        """Create a chat completion with optional tool calling (Cohere V2 format)."""
        # Validate model
        if model not in TINY_AYA_MODELS:
            raise ValueError(f"Model '{model}' not supported. Available: {list(TINY_AYA_MODELS.keys())}")

        # Convert messages to Cohere V2 format
        cohere_messages = self._convert_messages(messages)

        # Convert tools to Cohere V2 format
        cohere_tools = None
        if tools:
            cohere_tools = self._convert_tools(tools)

        client = self._get_client()

        payload = {
            "model": model,
            "messages": cohere_messages,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if cohere_tools:
            payload["tools"] = cohere_tools

        start = time.time()
        try:
            resp = client.chat(**payload)
            data = resp.model_dump()
        except Exception as e:
            error_str = str(e)
            if "rate limit" in error_str.lower() or "429" in error_str:
                raise RateLimitError("Rate limit exceeded. Please wait before retrying.")
            elif "401" in error_str or "unauthorized" in error_str.lower() or "authentication" in error_str.lower():
                raise AuthenticationError("Invalid API key.")
            elif "404" in error_str or "not found" in error_str.lower():
                raise ModelNotFoundError(f"Model '{model}' not found or not accessible.")
            raise

        elapsed_ms = int((time.time() - start) * 1000)
        return TinyAyaResponse(data, elapsed_ms)

    def _convert_messages(self, messages: List[Dict]) -> List[Any]:
        """Convert OpenAI-style messages to Cohere V2 format."""
        from cohere.types import (
            UserChatMessageV2,
            AssistantChatMessageV2,
            SystemChatMessageV2,
            ToolChatMessageV2,
        )

        cohere_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                cohere_messages.append(SystemChatMessageV2(content=content))
            elif role == "user":
                cohere_messages.append(UserChatMessageV2(content=content))
            elif role == "assistant":
                # Check for tool_calls in the message
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Convert tool_calls to Cohere format
                    tool_calls_v2 = []
                    for tc in tool_calls:
                        from cohere.types import ToolCallV2
                        tool_calls_v2.append(ToolCallV2(
                            id=tc.get("id", f"call_{int(time.time() * 1000)}"),
                            name=tc.get("name"),
                            parameters=tc.get("arguments", tc.get("parameters", {}))
                        ))
                    cohere_messages.append(AssistantChatMessageV2(
                        content=content,
                        tool_calls=tool_calls_v2
                    ))
                else:
                    cohere_messages.append(AssistantChatMessageV2(content=content))
            elif role == "tool":
                # Tool results in Cohere format
                tool_call_id = msg.get("tool_call_id", "")
                cohere_messages.append(ToolChatMessageV2(
                    tool_results=[{
                        "call": {"id": tool_call_id, "name": msg.get("name", "")},
                        "outputs": [{"type": "text", "text": content}]
                    }]
                ))
        return cohere_messages

    def _convert_tools(self, tools: List[Dict]) -> List[Any]:
        """Convert OpenAI-style tools to Cohere V2 format."""
        from cohere.types import ToolV2, ToolV2Function

        cohere_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                cohere_tools.append(ToolV2(
                    type="function",
                    function=ToolV2Function(
                        name=func.get("name"),
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    )
                ))
        return cohere_tools


class TinyAyaResponse:
    """Wrapper to match expected response interface for the investigation agent."""

    def __init__(self, data: dict, elapsed_ms: int):
        self.data = data
        self.elapsed_ms = elapsed_ms
        self.content = []

        # Cohere V2 response format: data.message with content and tool_calls
        message = data.get("message", {})

        # Handle text content
        if message.get("content"):
            for content_item in message["content"]:
                if content_item.get("type") == "text":
                    self.content.append(TextBlock(content_item["text"]))

        # Handle tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                self.content.append(ToolUseBlock(
                    id=tc.get("id", f"call_{int(time.time() * 1000)}"),
                    name=tc.get("name"),
                    input=tc.get("parameters", {}),
                ))

    @property
    def stop_reason(self) -> str:
        # Cohere V2 finish_reason
        finish = self.data.get("finish_reason", "stop")
        if finish == "tool_calls":
            return "tool_use"
        return "end_turn"


class TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class ToolUseBlock:
    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class ModelNotFoundError(Exception):
    """Raised when model is not found or accessible."""
    pass


def get_llm_client() -> TinyAyaClient:
    """Get the configured LLM client."""
    if not TINY_AYA_API_KEY:
        raise ValueError(
            "TINY_AYA_API_KEY is required but not configured. "
            "Set TINY_AYA_API_KEY in your .env file."
        )
    return TinyAyaClient()


def get_available_models() -> Dict[str, str]:
    """Get available Tiny Aya model variants."""
    return TINY_AYA_MODELS.copy()