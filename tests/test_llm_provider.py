import pytest

from swb.llm import LLMConfigError, normalize_model, resolve_provider


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ("SWB_PROVIDER", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_autodetect_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert resolve_provider() == "openai"


def test_autodetect_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert resolve_provider() == "openrouter"


def test_openai_wins_when_both_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    assert resolve_provider() == "openai"


def test_explicit_provider_overrides_autodetect(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    monkeypatch.setenv("SWB_PROVIDER", "openrouter")
    assert resolve_provider() == "openrouter"


def test_no_key_raises():
    with pytest.raises(LLMConfigError, match="No API key"):
        resolve_provider()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("SWB_PROVIDER", "anthropic")
    with pytest.raises(LLMConfigError, match="unknown SWB_PROVIDER"):
        resolve_provider()


def test_model_normalization_strips_prefix_for_openai():
    # Constitutions store the OpenRouter-style namespaced name.
    assert normalize_model("openai/gpt-4o-mini", "openai") == "gpt-4o-mini"
    assert normalize_model("gpt-4o-mini", "openai") == "gpt-4o-mini"


def test_model_normalization_adds_prefix_for_openrouter():
    assert normalize_model("gpt-4o-mini", "openrouter") == "openai/gpt-4o-mini"
    assert normalize_model("openai/gpt-4o-mini", "openrouter") == "openai/gpt-4o-mini"
    # A non-OpenAI vendor namespace must survive untouched.
    assert normalize_model("anthropic/claude-3.5-haiku", "openrouter") == "anthropic/claude-3.5-haiku"
