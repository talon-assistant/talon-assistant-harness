"""Multi-backend LLM API client.

Supports three API formats:
  koboldcpp  — POST /api/v1/generate  (KoboldCpp native API)
  llamacpp   — POST /completion       (llama.cpp server API)
  openai     — POST /v1/chat/completions  (OpenAI-compatible API)

The active format is set via ``config["llm"]["api_format"]`` (default:
``"koboldcpp"`` for backward compatibility).
"""

import requests
import json
from urllib.parse import urlparse


class LLMClient:
    """Multi-format LLM API wrapper for inference."""

    def __init__(self, config):
        llm_config = config["llm"]
        self.endpoint = llm_config["endpoint"]
        self.max_length = llm_config["max_length"]
        self.temperature = llm_config["temperature"]
        self.top_p = llm_config["top_p"]
        self.rep_pen = llm_config["rep_pen"]
        self.timeout = llm_config["timeout"]
        self.stop_sequences = llm_config["stop_sequences"]
        self.prompt_template = llm_config["prompt_template"]
        self.api_format = llm_config.get("api_format", "koboldcpp")
        # Optional reference to LLMServerManager (set by main.py in builtin mode).
        # Used to return a friendly "still loading" message instead of a 503/timeout.
        self.server_manager = None

    # ── Connection Test ───────────────────────────────────────

    def test_connection(self):
        """Test connection to the LLM server.

        Uses the appropriate method for each API format:
          koboldcpp  -> POST /api/v1/generate (short prompt)
          llamacpp   -> GET  /health
          openai     -> GET  /v1/models
        """
        try:
            if self.api_format == "llamacpp":
                return self._test_llamacpp()
            elif self.api_format == "openai":
                return self._test_openai()
            else:
                return self._test_koboldcpp()
        except Exception as e:
            print(f"   Warning: {e}")
            return False

    def _test_koboldcpp(self):
        """Test KoboldCpp connection with a minimal generation request."""
        test_response = requests.post(
            self.endpoint,
            json={
                "prompt": (
                    f"{self.prompt_template['user_prefix']}Hello"
                    f"{self.prompt_template['user_suffix']}"
                    f"{self.prompt_template['assistant_prefix']}"
                ),
                "max_length": 10,
                "stop_sequence": self.stop_sequences
            },
            timeout=10
        )
        if test_response.status_code == 200:
            print("   KoboldCpp connected!")
            return True
        else:
            print(f"   Warning: Status {test_response.status_code}")
            return False

    def _base_url(self) -> str:
        """Return the scheme+host+port of self.endpoint (no path).

        Used by health-check and model-list probes so they always hit the
        right host regardless of what path the endpoint is configured with.
        """
        parsed = urlparse(self.endpoint)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _test_llamacpp(self):
        """Test llama.cpp server via /health endpoint."""
        health_url = f"{self._base_url()}/health"
        try:
            resp = requests.get(health_url, timeout=10)
        except requests.ConnectionError:
            print(f"   Warning: llama.cpp server not reachable at {health_url}")
            return False
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "")
            if status == "ok":
                print("   llama.cpp server connected!")
                return True
            if status == "loading":
                print("   llama.cpp server is loading model — will retry at request time")
                return True  # Not an error; server will be ready soon
        print(f"   Warning: llama.cpp health check returned {resp.status_code}")
        return False

    def _test_openai(self):
        """Test OpenAI-compatible server via /v1/models endpoint."""
        models_url = f"{self._base_url()}/v1/models"
        try:
            resp = requests.get(models_url, timeout=10)
        except requests.ConnectionError:
            print(f"   Warning: OpenAI server not reachable at {models_url}")
            return False
        if resp.status_code == 200:
            print("   OpenAI-compatible server connected!")
            return True
        print(f"   Warning: OpenAI models endpoint returned {resp.status_code}")
        return False

    # ── Generation ────────────────────────────────────────────

    def generate(self, prompt, use_vision=False, screenshot_b64=None,
                 images_b64=None, max_length=None, system_prompt=None,
                 temperature=None):
        """Send prompt to LLM server and return generated text.

        Dispatches to the appropriate backend based on ``self.api_format``.

        Args:
            prompt: The user message / content to send.
            use_vision: Whether this is a vision request.
            screenshot_b64: Single base64 image (legacy; merged into images_b64).
            images_b64: List of base64-encoded PNG strings (multi-image support).
            max_length: Override max token generation length.
            system_prompt: Optional system message (ChatML <|im_start|>system).
            temperature: Override temperature for this call.
        """
        # Normalise: merge legacy screenshot_b64 into images_b64 list.
        effective_images = list(images_b64) if images_b64 else []
        if screenshot_b64 and screenshot_b64 not in effective_images:
            effective_images.insert(0, screenshot_b64)
        effective_images = effective_images or None

        if self.api_format == "llamacpp":
            return self._generate_llamacpp(
                prompt, use_vision, effective_images,
                max_length, system_prompt, temperature)
        elif self.api_format == "openai":
            return self._generate_openai(
                prompt, use_vision, effective_images,
                max_length, system_prompt, temperature)
        else:
            return self._generate_koboldcpp(
                prompt, use_vision, effective_images,
                max_length, system_prompt, temperature)

    # ── KoboldCpp Backend ─────────────────────────────────────

    def _generate_koboldcpp(self, prompt, use_vision=False, images_b64=None,
                            max_length=None, system_prompt=None, temperature=None):
        """KoboldCpp native API: POST /api/v1/generate."""
        effective_max_length = max_length or self.max_length
        effective_temperature = (temperature if temperature is not None
                                 else self.temperature)

        formatted_prompt = self._build_chatml_prompt(
            prompt, system_prompt, use_vision)

        payload = {
            "prompt": formatted_prompt,
            "max_length": effective_max_length,
            "temperature": effective_temperature,
            "top_p": self.top_p,
            "rep_pen": self.rep_pen,
            "stop_sequence": self.stop_sequences
        }

        if use_vision and images_b64:
            payload["images"] = images_b64

        try:
            response = requests.post(
                self.endpoint, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                if result and 'results' in result and len(result['results']) > 0:
                    return result['results'][0]['text'].strip()
                else:
                    return "Error: Unexpected response format"
            else:
                return f"Error: Status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    # ── llama.cpp Backend ─────────────────────────────────────

    def _generate_llamacpp(self, prompt, use_vision=False, images_b64=None,
                           max_length=None, system_prompt=None, temperature=None):
        """llama.cpp server API: POST /completion."""
        # If we have a server manager reference, check readiness before sending.
        # Returns a friendly message instead of hanging/503 while model loads.
        if self.server_manager is not None:
            status = self.server_manager.status
            if status == "starting":
                return "I'm still loading the language model. Please try again in a moment."
            if status == "error":
                return "Error: LLM server failed to start. Check the LLM Server settings."
            if status == "stopped":
                return "Error: LLM server is not running."

        effective_max_length = max_length or self.max_length
        effective_temperature = (temperature if temperature is not None
                                 else self.temperature)

        formatted_prompt = self._build_chatml_prompt(
            prompt, system_prompt, use_vision)

        payload = {
            "prompt": formatted_prompt,
            "n_predict": effective_max_length,
            "temperature": effective_temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.rep_pen,
            "stop": self.stop_sequences,
            "stream": False,
        }

        if use_vision and images_b64:
            payload["image_data"] = [
                {"data": b64, "id": 10 + i} for i, b64 in enumerate(images_b64)
            ]

        try:
            response = requests.post(
                self.endpoint, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                content = result.get("content", "")
                if content:
                    return content.strip()
                return result.get("text", "Error: Unexpected response format").strip()
            else:
                return f"Error: Status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    # ── OpenAI-Compatible Backend ─────────────────────────────

    def _generate_openai(self, prompt, use_vision=False, images_b64=None,
                         max_length=None, system_prompt=None, temperature=None):
        """OpenAI-compatible API: POST /v1/chat/completions."""
        effective_max_length = max_length or self.max_length
        effective_temperature = (temperature if temperature is not None
                                 else self.temperature)

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if use_vision and images_b64:
            content = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": effective_max_length,
            "temperature": effective_temperature,
            "top_p": self.top_p,
            "stop": self.stop_sequences,
            "stream": False,
        }

        try:
            response = requests.post(
                self.endpoint, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content", "").strip()
                return "Error: No choices in response"
            else:
                return f"Error: Status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    # ── Prompt Building ───────────────────────────────────────

    def _build_chatml_prompt(self, prompt, system_prompt=None,
                             use_vision=False):
        """Build a ChatML-formatted prompt string.

        Used by koboldcpp and llamacpp backends (which take raw prompt text).
        The openai backend builds messages directly instead.
        """
        system_block = ""
        if system_prompt:
            system_block = (
                f"<|im_start|>system\n"
                f"{system_prompt}<|im_end|>\n"
            )

        if use_vision:
            vision_prefix = self.prompt_template.get("vision_prefix", "")
            formatted = (
                f"{system_block}"
                f"{self.prompt_template['user_prefix']}"
                f"{vision_prefix}{prompt}"
                f"{self.prompt_template['user_suffix']}"
                f"{self.prompt_template['assistant_prefix']}"
            )
        else:
            formatted = (
                f"{system_block}"
                f"{self.prompt_template['user_prefix']}"
                f"{prompt}"
                f"{self.prompt_template['user_suffix']}"
                f"{self.prompt_template['assistant_prefix']}"
            )

        # Diagnostic: log prompt size so context-window issues are visible
        char_count = len(formatted)
        approx_tokens = char_count // 4  # rough estimate: ~4 chars per token
        print(f"   [LLM] Prompt size: {char_count} chars (~{approx_tokens} tokens)")

        return formatted
