"""LLM client factory — supports OpenAI and OpenRouter.

Every LLM call in the benchmark (peer simulator, eval agent, fuzzy judge) goes through
here, so the provider is configured in exactly one place. Both providers speak the
OpenAI wire format, so the same `openai` SDK serves both; only the base URL and the
API key env var differ.

Provider selection, in priority order:
  1. SWB_PROVIDER env var ("openai" | "openrouter"), if set
  2. auto-detect: whichever of OPENAI_API_KEY / OPENROUTER_API_KEY is present
     (OPENAI_API_KEY wins if both are set)

Model names are provider-specific: OpenAI wants "gpt-4o-mini"; OpenRouter wants the
namespaced "openai/gpt-4o-mini". `normalize_model` translates between the two so a
world's constitution can be run against either provider unchanged.
"""

from __future__ import annotations

import os

PROVIDERS = {
    "openai": {
        "base_url": None,  # SDK default (https://api.openai.com/v1)
        "key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
}


class LLMConfigError(RuntimeError):
    pass


def resolve_provider() -> str:
    """Pick the provider from SWB_PROVIDER, else from whichever key is present."""
    explicit = os.environ.get("SWB_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise LLMConfigError(
                f"unknown SWB_PROVIDER {explicit!r}; expected one of {sorted(PROVIDERS)}"
            )
        return explicit
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    raise LLMConfigError(
        "No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
        "(optionally pin the provider with SWB_PROVIDER)."
    )


def normalize_model(model: str, provider: str | None = None) -> str:
    """Make a model name valid for the active provider.

    OpenRouter namespaces models ("openai/gpt-4o-mini"); the OpenAI API does not
    ("gpt-4o-mini"). Constitutions store the namespaced form, so strip the vendor
    prefix when talking to OpenAI directly, and add it back for OpenRouter.
    """
    provider = provider or resolve_provider()
    if provider == "openai":
        return model.split("/", 1)[1] if model.startswith("openai/") else model
    if provider == "openrouter" and "/" not in model:
        return f"openai/{model}"
    return model


def get_client():
    """Return an OpenAI SDK client pointed at the active provider."""
    provider = resolve_provider()
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise LLMConfigError(
            f"{cfg['key_env']} is not set (provider={provider}). "
            "Required for --mode llm and for `swb eval`."
        )
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def chat(messages: list[dict], model: str, temperature: float = 0.7, tools: list | None = None):
    """One chat completion. Returns the raw SDK message object."""
    provider = resolve_provider()
    client = get_client()
    kwargs: dict = {
        "model": normalize_model(model, provider),
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message
