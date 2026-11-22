"""Client for open-weight models served behind an OpenAI-compatible API.

Works with vLLM, SGLang, Ollama, llama.cpp server, LM Studio, TGI, and hosted
open-weight endpoints (Together, Fireworks, OpenRouter, ...). Unlike the
``openai`` provider it does not rely on OpenAI-only features (strict
``.parse()``, ``max_completion_tokens``, ``reasoning_effort``) and it is more
forgiving about what the model returns (think blocks, markdown fences).
"""

import json
import re
import threading
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Literal, cast, overload

import pydantic
from openai import AsyncOpenAI

from ..base import (
    AllowedChatCompletionMessageParams,
    ProviderClient,
    TResponseModel,
    Usage,
)
from ..config import BaseLLMConfig, EnvField

StructuredMode = Literal["json_schema", "tools", "json_object", "prompt"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LocalConfig(BaseLLMConfig):
    """Configuration for a self-hosted / open-weight OpenAI-compatible server."""

    provider: Literal["local"] = EnvField("LLM_PROVIDER", default="local")  # pyright: ignore[reportIncompatibleVariableOverride]
    base_url: str = EnvField("LOCAL_LLM_BASE_URL", default="http://localhost:8000/v1")
    # Most local servers ignore the key, but the OpenAI SDK requires one.
    api_key: str = EnvField("LOCAL_LLM_API_KEY", default="EMPTY", exclude=True)
    # How to obtain structured output. json_schema = guided decoding (best on
    # vLLM/SGLang/Ollama); tools = forced function call; json_object = JSON mode
    # with schema in the prompt; prompt = schema in prompt only.
    structured_mode: StructuredMode = EnvField(
        "LOCAL_LLM_STRUCTURED_MODE", default="json_schema", exclude=True
    )
    request_timeout: float = EnvField(
        "LOCAL_LLM_TIMEOUT", default=600.0, exclude=True
    )


def _clean(text: str) -> str:
    """Strip reasoning blocks and markdown fences that open models often emit."""
    text = _THINK_RE.sub("", text)
    # Unterminated think block (hit max_tokens mid-reasoning or server parser
    # removed only the opening tag): keep what comes after the last </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return _FENCE_RE.sub("", text.strip()).strip()


def _extract_json(text: str) -> str:
    """Return the outermost JSON object in text (tolerates leading chatter)."""
    text = _clean(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class LocalClient(ProviderClient[LocalConfig]):
    """Client for open-weight models behind an OpenAI-compatible endpoint."""

    _client_cache: dict[str, "LocalClient"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: LocalConfig | None = None):
        """Initialize the client from config or environment."""
        config = LocalConfig() if config is None else LocalConfig.model_validate(config)
        super().__init__(config)
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
        )

    @staticmethod
    def _get_cache_key(config: LocalConfig) -> str:
        # base_url must be part of the key so several servers can coexist.
        payload = json.dumps(
            [config.provider, config.base_url, config.api_key, config.structured_mode]
        )
        return sha256(payload.encode()).hexdigest()

    @staticmethod
    def from_cache(config: LocalConfig) -> "LocalClient":
        """Get or create a cached client for this config."""
        key = LocalClient._get_cache_key(config)
        with LocalClient._cache_lock:
            if key not in LocalClient._client_cache:
                LocalClient._client_cache[key] = LocalClient(config)
            return LocalClient._client_cache[key]

    @overload
    async def _generate(
        self,
        *,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,
        response_format: None = None,
        **kwargs: Any,
    ) -> tuple[str, Usage]: ...

    @overload
    async def _generate(
        self,
        *,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,
        response_format: type[TResponseModel],
        **kwargs: Any,
    ) -> tuple[TResponseModel, Usage]: ...

    async def _generate(
        self,
        *,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,  # unused: not portable
        response_format: type[TResponseModel] | None = None,
        **kwargs: Any,
    ) -> tuple[str, Usage] | tuple[TResponseModel, Usage]:
        """Generate a completion against the OpenAI-compatible server."""
        args: dict[str, Any] = {"model": model, **kwargs}
        args.pop("max_completion_tokens", None)
        if max_tokens is not None:
            args["max_tokens"] = max_tokens
        if temperature is not None:
            args["temperature"] = temperature

        # Copy so retries don't mutate the caller's (and the logged) prompt.
        msgs: list[dict[str, Any]] = [dict(m) for m in messages]

        if response_format is None:
            content, tokens = await self._chat(msgs, args)
            return _clean(content), self._usage(model, tokens)

        return await self._generate_structured(model, msgs, args, response_format)

    async def _chat(
        self, msgs: list[dict[str, Any]], args: dict[str, Any]
    ) -> tuple[str, int]:
        resp = await self.client.chat.completions.create(messages=cast(Any, msgs), **args)
        tokens = resp.usage.total_tokens if resp.usage else 0
        content = resp.choices[0].message.content if resp.choices else None
        return content or "", tokens

    async def _generate_structured(
        self,
        model: str,
        msgs: list[dict[str, Any]],
        args: dict[str, Any],
        response_format: type[TResponseModel],
    ) -> tuple[TResponseModel, Usage]:
        schema = response_format.model_json_schema()
        name = response_format.__name__
        mode = self.config.structured_mode
        total_tokens = 0
        errors: list[str] = []

        if mode in ("json_object", "prompt"):
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Respond with a single JSON object and nothing else. It must "
                        f"validate against this JSON schema:\n{json.dumps(schema)}"
                    ),
                },
                *msgs,
            ]

        for _ in range(3):
            call_args = dict(args)
            if mode == "json_schema":
                call_args["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": schema},
                }
            elif mode == "json_object":
                call_args["response_format"] = {"type": "json_object"}
            elif mode == "tools":
                call_args["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": f"Generate{name}",
                            "description": f"Generate a {name}.",
                            "parameters": schema,
                        },
                    }
                ]
                call_args["tool_choice"] = "required"

            resp = await self.client.chat.completions.create(
                messages=cast(Any, msgs), **call_args
            )
            total_tokens += resp.usage.total_tokens if resp.usage else 0
            if not resp.choices:
                errors.append("empty choices")
                continue
            message = resp.choices[0].message

            if mode == "tools" and message.tool_calls:
                raw = message.tool_calls[0].function.arguments  # type: ignore[union-attr]
            else:
                raw = message.content or ""

            try:
                parsed = response_format.model_validate_json(_extract_json(raw))
                return parsed, self._usage(model, total_tokens)
            except (json.JSONDecodeError, pydantic.ValidationError) as e:
                errors.append(str(e))
                msgs.append({"role": "assistant", "content": raw})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            f"That output did not match the required {name} schema:\n"
                            f"{e}\nReturn only a corrected JSON object."
                        ),
                    }
                )

        raise RuntimeError(
            f"Exceeded attempts to parse {name} from {model}. Errors: "
            + " -> ".join(errors)
        )

    def _usage(self, model: str, tokens: int) -> Usage:
        return Usage(token_count=tokens, provider="local", model=model)
