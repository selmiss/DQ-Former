import importlib
import os
from types import SimpleNamespace

import pytest


chatgpt_module = importlib.import_module("evaluation.chatgpt_hallu_eval")


def test_extract_final_answer_text_variants():
    extract = chatgpt_module.extract_final_answer_text
    assert extract("Final answer: positive") == "positive"
    assert extract("Random text") == "Random text"
    assert extract("") == ""


def test_build_user_prompt_contains_inputs():
    prompt = chatgpt_module.build_user_prompt("C1=CC=CC=C1", "hydroxyl, amide", "Final answer: hydroxyl")
    assert "C1=CC=CC=C1" in prompt
    assert "hydroxyl, amide" in prompt
    assert "Final answer: hydroxyl" in prompt


def test_parse_chatgpt_json_handles_code_fence():
    fenced = """```json
    {"hallucination": false, "extra_functional_groups": []}
    ```"""
    parsed = chatgpt_module.parse_chatgpt_json(fenced)
    assert parsed["hallucination"] is False
    assert parsed["extra_functional_groups"] == []


def test_coerce_bool_accepts_common_representations():
    coerce = chatgpt_module.coerce_bool
    assert coerce(True) is True
    assert coerce("true") is True
    assert coerce("No") is False
    assert coerce(1) is True
    assert coerce(0.0) is False
    assert coerce("maybe") is None


def test_request_chatgpt_review_returns_message():
    dummy_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=' {"hallucination": false} '))]
    )

    class DummyCompletions:
        @staticmethod
        def create(**_kwargs):
            return dummy_completion

    chatgpt_module.OPENAI_CLIENT = SimpleNamespace(
        chat=SimpleNamespace(completions=DummyCompletions())
    )
    chatgpt_module.OPENAI_CLIENT_KIND = "client"

    args = SimpleNamespace(
        chatgpt_model="fake-model",
        chatgpt_temperature=0,
        chatgpt_max_retries=1,
        chatgpt_retry_backoff=0.0,
        openai_api_key=None,
    )

    try:
        content = chatgpt_module.request_chatgpt_review("prompt", args)
        assert content.strip() == '{"hallucination": false}'
    finally:
        chatgpt_module.OPENAI_CLIENT = None
        chatgpt_module.OPENAI_CLIENT_KIND = None


def test_ensure_openai_client_sets_api_key(monkeypatch):
    stub_openai = SimpleNamespace(api_key=None)
    monkeypatch.setattr(chatgpt_module, "openai", stub_openai, raising=False)
    chatgpt_module.OPENAI_CLIENT = None
    chatgpt_module.OPENAI_CLIENT_KIND = None

    chatgpt_module.ensure_openai_client("test-key")
    assert stub_openai.api_key == "test-key"
    assert chatgpt_module.OPENAI_CLIENT is stub_openai
    assert chatgpt_module.OPENAI_CLIENT_KIND == "legacy"

    chatgpt_module.OPENAI_CLIENT = None
    chatgpt_module.OPENAI_CLIENT_KIND = None


def test_request_chatgpt_review_live_with_env_key():
    """Integration test that hits the real ChatGPT API using env credentials."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping live ChatGPT integration test.")

    chatgpt_module.OPENAI_CLIENT = None
    chatgpt_module.OPENAI_CLIENT_KIND = None
    chatgpt_module.ensure_openai_client(None)
    args = SimpleNamespace(
        chatgpt_model=os.environ.get("OPENAI_EVAL_MODEL", "gpt-4.1-mini"),
        chatgpt_temperature=0.0,
        chatgpt_max_retries=2,
        chatgpt_retry_backoff=1.0,
        openai_api_key=None,
    )

    user_prompt = chatgpt_module.build_user_prompt(
        smiles="CCO",
        ground_truth="alcohol",
        model_output="Final answer: alcohol",
    )

    content = chatgpt_module.request_chatgpt_review(user_prompt, args)
    assert content and content.strip(), "ChatGPT API returned empty content"
    parsed = chatgpt_module.parse_chatgpt_json(content)
    assert parsed is not None, "ChatGPT API response not valid JSON"
    assert "hallucination" in parsed

