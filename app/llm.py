"""LLM helpers. Works with any OpenAI-compatible endpoint.

structured() asks for JSON, validates with pydantic and retries once with the
validation error.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

# provider quota throttle
# Providers with strict per-minute quotas (e.g. the Gemini free tier allows 5
# requests/minute) cannot absorb the workflow's parallel bursts. When
# configured, calls are spaced so at most `llm_max_requests_per_minute` start
# per rolling minute.
_rate_lock = asyncio.Lock()
_last_call_at = 0.0


async def _throttle(settings: Settings) -> None:
    limit = settings.llm_max_requests_per_minute
    if limit <= 0:
        return
    global _last_call_at
    interval = 60.0 / limit
    async with _rate_lock:
        wait = _last_call_at + interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_at = time.monotonic()


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def get_chat_model(settings: Settings | None = None, temperature: float | None = None) -> BaseChatModel:
    """Return the configured chat model."""
    from langchain_openai import ChatOpenAI

    settings = settings or get_settings()
    # Not every provider accepts reasoning_effort, set it to "omit" to skip it.
    effort = settings.llm_reasoning_effort
    extra_body = None if effort.strip().lower() in ("", "omit") else {"reasoning_effort": effort}
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.openai_base_url,
        api_key=settings.resolved_api_key or "missing-key",
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        extra_body=extra_body,
    )


async def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: BaseChatModel | None = None,
    temperature: float | None = None,
) -> str:
    """Plain text completion."""
    chat = model or get_chat_model(temperature=temperature)
    messages = [SystemMessage(content=system)] if system else []
    messages.append(HumanMessage(content=prompt))
    await _throttle(get_settings())
    response = await chat.ainvoke(messages)
    text = response.text  # langchain-core >=1.6: property; older/fakes: method
    return (text() if callable(text) else text or "").strip()


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model response")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : index + 1])
                if isinstance(parsed, dict):
                    return parsed
                break
    raise ValueError("no JSON object found in model response")


async def structured(
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    model: BaseChatModel | None = None,
    temperature: float | None = None,
    max_attempts: int = 2,
) -> T:
    """Ask for JSON matching `schema`, validate, retry once with feedback."""
    chat = model or get_chat_model(temperature=temperature)
    instruction = (
        f"{prompt}\n\n"
        "Respond with a single JSON object only - no prose, no markdown fences, "
        f"matching this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    )
    last_error = ""
    for attempt in range(max_attempts):
        attempt_prompt = instruction if attempt == 0 else (
            f"{prompt}\n\nYour previous response could not be parsed: {last_error}\n"
            "Respond with a single valid JSON object only, matching this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        raw = await complete(attempt_prompt, system=system, model=chat, temperature=temperature)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:600]
    raise ValueError(f"model could not produce valid {schema.__name__}: {last_error}")