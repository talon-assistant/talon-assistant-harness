"""Tests for core.llm_client module."""

import json
import pytest
from unittest.mock import patch, MagicMock

from core.llm_client import LLMClient, LLMError


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(api_format="koboldcpp"):
    return {
        "llm": {
            "endpoint": "http://localhost:5001/api/v1/generate",
            "max_length": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "rep_pen": 1.1,
            "timeout": 30,
            "stop_sequences": ["<|im_end|>"],
            "prompt_template": {
                "user_prefix": "<|im_start|>user\n",
                "user_suffix": "<|im_end|>\n",
                "assistant_prefix": "<|im_start|>assistant\n",
                "vision_prefix": "[img-1]\n",
            },
            "api_format": api_format,
        }
    }


def _make_client(api_format="koboldcpp"):
    return LLMClient(_make_config(api_format))


# ── LLMError ─────────────────────────────────────────────────────────────────

def test_llm_error_with_status_code():
    err = LLMError("server error", status_code=500)
    assert str(err) == "server error"
    assert err.status_code == 500


def test_llm_error_without_status_code():
    err = LLMError("generic error")
    assert str(err) == "generic error"
    assert err.status_code is None


# ── _build_chatml_prompt() ───────────────────────────────────────────────────

def test_build_chatml_prompt_basic():
    client = _make_client()
    result = client._build_chatml_prompt("hello")
    assert "<|im_start|>user\nhello<|im_end|>" in result
    assert "<|im_start|>assistant\n" in result


def test_build_chatml_prompt_with_system():
    client = _make_client()
    result = client._build_chatml_prompt("hello", system_prompt="You are helpful.")
    assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
    assert "<|im_start|>user\nhello" in result


def test_build_chatml_prompt_with_vision():
    client = _make_client()
    result = client._build_chatml_prompt("describe this", use_vision=True)
    assert "[img-1]\n" in result


# ── generate() KoboldCpp ────────────────────────────────────────────────────

@patch("core.llm_client.requests.post")
def test_generate_koboldcpp_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"text": " Hello there! "}]}
    mock_post.return_value = mock_resp

    client = _make_client("koboldcpp")
    result = client.generate("hi")
    assert result == "Hello there!"


@patch("core.llm_client.requests.post")
def test_generate_koboldcpp_http_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_post.return_value = mock_resp

    client = _make_client("koboldcpp")
    with pytest.raises(LLMError) as exc_info:
        client.generate("hi")
    assert exc_info.value.status_code == 500


@patch("core.llm_client.requests.post")
def test_generate_koboldcpp_connection_error(mock_post):
    import requests as req
    mock_post.side_effect = req.ConnectionError("refused")

    client = _make_client("koboldcpp")
    with pytest.raises(LLMError):
        client.generate("hi")


# ── generate() llama.cpp ────────────────────────────────────────────────────

@patch("core.llm_client.requests.post")
def test_generate_llamacpp_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"content": " Some answer "}
    mock_post.return_value = mock_resp

    client = _make_client("llamacpp")
    result = client.generate("hi")
    assert result == "Some answer"


def test_generate_llamacpp_server_starting():
    """When server is still loading, return friendly message instead of error."""
    client = _make_client("llamacpp")
    mock_mgr = MagicMock()
    mock_mgr.status = "starting"
    client.server_manager = mock_mgr
    result = client.generate("hi")
    assert "loading" in result.lower() or "try again" in result.lower()


def test_generate_llamacpp_server_stopped():
    client = _make_client("llamacpp")
    mock_mgr = MagicMock()
    mock_mgr.status = "stopped"
    client.server_manager = mock_mgr
    with pytest.raises(LLMError):
        client.generate("hi")


# ── generate() OpenAI ───────────────────────────────────────────────────────

@patch("core.llm_client.requests.post")
def test_generate_openai_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": " OpenAI answer "}}]
    }
    mock_post.return_value = mock_resp

    client = _make_client("openai")
    result = client.generate("hi")
    assert result == "OpenAI answer"


@patch("core.llm_client.requests.post")
def test_generate_openai_empty_choices(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}
    mock_post.return_value = mock_resp

    client = _make_client("openai")
    with pytest.raises(LLMError, match="No choices"):
        client.generate("hi")


# ── test_connection() ────────────────────────────────────────────────────────

@patch("core.llm_client.requests.post")
def test_connection_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"text": "ok"}]}
    mock_post.return_value = mock_resp

    client = _make_client("koboldcpp")
    assert client.test_connection() is True


@patch("core.llm_client.requests.post")
def test_connection_failure(mock_post):
    import requests as req
    mock_post.side_effect = req.ConnectionError("refused")

    client = _make_client("koboldcpp")
    assert client.test_connection() is False
