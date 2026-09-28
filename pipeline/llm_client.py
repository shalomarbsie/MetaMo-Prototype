from __future__ import annotations

import importlib
import os
import time
from typing import Any


def _build_openai_default_headers() -> dict[str, str]:
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    app_title = os.getenv("OPENROUTER_X_TITLE", "").strip()
    headers: dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-OpenRouter-Title"] = app_title
    return headers


def build_chat_llm(
    *,
    provider_name: str,
    model: str,
    temperature: float,
    api_key: str,
) -> Any:
    if provider_name == "openai":
        openai_mod = importlib.import_module("langchain_openai")
        ChatOpenAI = getattr(openai_mod, "ChatOpenAI")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        default_headers = _build_openai_default_headers()

        openai_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
        }
        if base_url:
            openai_kwargs["base_url"] = base_url
        if default_headers:
            openai_kwargs["default_headers"] = default_headers

        return ChatOpenAI(**openai_kwargs)

    if provider_name == "gemini":
        genai_mod = importlib.import_module("langchain_google_genai")
        ChatGoogleGenerativeAI = getattr(genai_mod, "ChatGoogleGenerativeAI")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )

    raise RuntimeError(f"Unsupported provider '{provider_name}'")


def check_llm_health(
    *,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    from dotenv import load_dotenv
    from utils import resolve_provider_and_model_name

    load_dotenv()
    resolved_provider, resolved_model = resolve_provider_and_model_name(
        explicit_provider_name=provider_name
    )
    active_provider = provider_name or resolved_provider
    active_model = model_name or resolved_model

    if active_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return {
                "ok": False,
                "provider": active_provider,
                "model": active_model,
                "latency_ms": None,
                "error": "OPENAI_API_KEY is missing or empty in environment",
            }
    elif active_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {
                "ok": False,
                "provider": active_provider,
                "model": active_model,
                "latency_ms": None,
                "error": "GEMINI_API_KEY is missing or empty in environment",
            }
    else:
        return {
            "ok": False,
            "provider": active_provider,
            "model": active_model,
            "latency_ms": None,
            "error": f"Unsupported provider '{active_provider}'",
        }

    start_time = time.perf_counter()
    try:
        from langchain_core.messages import HumanMessage

        llm = build_chat_llm(
            provider_name=active_provider,
            model=active_model,
            temperature=0.0,
            api_key=api_key,
        )
        llm.invoke([HumanMessage(content="ping")])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "ok": True,
            "provider": active_provider,
            "model": active_model,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        error_msg = str(exc)
        if api_key and api_key in error_msg:
            error_msg = error_msg.replace(api_key, "[REDACTED]")
        return {
            "ok": False,
            "provider": active_provider,
            "model": active_model,
            "latency_ms": latency_ms,
            "error": error_msg,
        }
