"""Multi-backend LLM API client.

Supports three API formats:
  koboldcpp  — POST /api/v1/generate  (KoboldCpp native API)
  llamacpp   — POST /completion       (llama.cpp server API)
  openai     — POST /v1/chat/completions  (OpenAI-compatible API)

The active format is set via ``config["llm"]["api_format"]`` (default:
``"koboldcpp"`` for backward compatibility).
"""

import re
import requests
import json
from urllib.parse import urlparse

import logging
log = logging.getLogger(__name__)


# Qwen 3+ and similar reasoning models wrap their chain-of-thought in
# <think>...</think> or <thinking>...</thinking> blocks. These tags MUST
# be stripped before any downstream consumer (user display, JSON parser,
# regex extractor, voice synthesis) sees the output. Otherwise tags and
# reasoning content leak into UIs, break argument parsing ("location =
# '<think>'"), and confuse tool-call extractors.
#
# Handles:
#  - Matched pairs: <think>...</think>
#  - Unclosed opening tag: <think>... (take everything before as answer,
#    or if answer is empty, drop the whole trailing think block)
#  - Stray closing tag: ...</think>content — keep content after the tag
#  - Both <think> and <thinking> variants
#  - Multiline content inside tags
_THINK_TAG_PATTERNS = [
    re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
]
_UNCLOSED_THINK_PATTERN = re.compile(
    r'<think(?:ing)?>.*$', re.DOTALL | re.IGNORECASE)
_STRAY_CLOSE_PATTERN = re.compile(
    r'^.*?</think(?:ing)?>\s*', re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove <think>/<thinking> reasoning blocks from LLM output.

    Returns the final answer with all reasoning blocks removed. Called
    on every generate() return so downstream consumers never see the
    reasoning tokens.
    """
    if not text or "<think" not in text.lower() and "</think" not in text.lower():
        return text

    cleaned = text
    # Remove matched pairs first
    for pattern in _THINK_TAG_PATTERNS:
        cleaned = pattern.sub('', cleaned)

    # Handle unclosed <think>... (model got cut off mid-reasoning).
    # If there's content AFTER the opening tag to end-of-string and nothing
    # closes it, the "answer" is whatever came before the tag.
    if re.search(r'<think(?:ing)?>', cleaned, re.IGNORECASE):
        before = re.split(
            r'<think(?:ing)?>', cleaned, maxsplit=1,
            flags=re.IGNORECASE)[0]
        cleaned = before

    # Handle stray closing tag without matching opener: assume everything
    # before the closing tag was reasoning, keep content after.
    if re.search(r'</think(?:ing)?>', cleaned, re.IGNORECASE):
        after_split = re.split(
            r'</think(?:ing)?>', cleaned, maxsplit=1,
            flags=re.IGNORECASE)
        if len(after_split) == 2:
            cleaned = after_split[1]

    return cleaned.strip()


class LLMError(Exception):
    """Raised when an LLM generation request fails."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _truncate_degeneration(text: str, *, ngram_size: int = 5,
                           repeat_threshold: int = 3) -> str:
    """Detect and truncate LLM degeneration loops.

    Two detection strategies:
    1. **N-gram repetition**: scans for repeated 5-grams. Catches copy-paste loops.
    2. **Run-on detection**: catches semantic-loop degeneration where the model
       produces a wall of text without sentence breaks. If any stretch exceeds
       60 words without a sentence-ending punctuation mark (. ! ? or newline),
       the text is truncated at the last complete sentence.

    Returns the (possibly shortened) text.
    """
    # ── Strategy 1: N-gram repetition ────────────────────────
    words = text.split()
    if len(words) >= ngram_size * repeat_threshold:
        seen: dict[tuple, list[int]] = {}  # ngram -> list of word-positions
        for i in range(len(words) - ngram_size + 1):
            gram = tuple(w.lower() for w in words[i:i + ngram_size])
            positions = seen.setdefault(gram, [])
            positions.append(i)
            if len(positions) >= repeat_threshold:
                # Truncate at the start of the second occurrence
                cut = positions[1]
                truncated = " ".join(words[:cut]).rstrip(" ,;:-")
                if len(truncated) > 40:
                    log.warning(
                        f"[LLM] Degeneration detected (ngram repeated "
                        f"{len(positions)}x at word {cut}/{len(words)}), "
                        f"truncating"
                    )
                    return truncated

    # ── Strategy 2: Run-on sentence detection ────────────────
    # Split on sentence-ending punctuation. If any segment has 60+
    # words, the model has entered a semantic degeneration spiral
    # (unique vocabulary but no coherent structure).
    import re as _re
    RUN_ON_THRESHOLD = 60
    sentences = _re.split(r'(?<=[.!?\n])\s+', text)
    if len(sentences) > 1:
        for idx, sent in enumerate(sentences):
            word_count = len(sent.split())
            if word_count >= RUN_ON_THRESHOLD:
                # Keep everything up to (but not including) this run-on sentence
                good_part = ". ".join(sentences[:idx]).rstrip(" ,;:-")
                if not good_part:
                    # The very first sentence is a run-on — take first 60 words
                    good_part = " ".join(sent.split()[:RUN_ON_THRESHOLD])
                good_part = good_part.rstrip(" ,;:-")
                if good_part and len(good_part) > 40:
                    log.warning(
                        f"[LLM] Run-on degeneration detected (segment {idx} "
                        f"has {word_count} words without sentence break), "
                        f"truncating"
                    )
                    return good_part

    return text


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
        # Thinking model support: when True, /no_think and /think directives
        # are injected into system prompts based on call-site hints, and
        # max_length is boosted when thinking is active. Auto-detected on
        # first generate() call (see _ensure_thinking_detected).
        self.thinking_boost = float(
            llm_config.get("thinking_max_length_boost", 2.5))
        self._is_thinking_model: bool | None = None
        self._thinking_probe_attempted: bool = False
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
            log.warning(f"Warning: {e}")
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
            log.info("KoboldCpp connected!")
            return True
        else:
            log.warning(f"Warning: Status {test_response.status_code}")
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
            log.warning(f"Warning: llama.cpp server not reachable at {health_url}")
            return False
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "")
            if status == "ok":
                log.info("llama.cpp server connected!")
                return True
            if status == "loading":
                log.info("llama.cpp server is loading model — will retry at request time")
                return True  # Not an error; server will be ready soon
        log.warning(f"Warning: llama.cpp health check returned {resp.status_code}")
        return False

    def _test_openai(self):
        """Test OpenAI-compatible server via /v1/models endpoint."""
        models_url = f"{self._base_url()}/v1/models"
        try:
            resp = requests.get(models_url, timeout=10)
        except requests.ConnectionError:
            log.warning(f"Warning: OpenAI server not reachable at {models_url}")
            return False
        if resp.status_code == 200:
            log.info("OpenAI-compatible server connected!")
            return True
        log.warning(f"Warning: OpenAI models endpoint returned {resp.status_code}")
        return False

    # ── Generation ────────────────────────────────────────────

    def _ensure_thinking_detected(self):
        """Probe the loaded model ONCE to see if it emits <think> tags.

        Runs a tiny inference call and checks the output for <think>.
        Result is cached for the rest of the session. Safe to call
        repeatedly — only actually probes once.

        Fails gracefully: if KoboldCpp isn't reachable, assumes
        non-thinking model and moves on.
        """
        if self._thinking_probe_attempted:
            return
        self._thinking_probe_attempted = True
        try:
            # Direct backend call so we don't recurse through generate()
            if self.api_format == "llamacpp":
                raw = self._generate_llamacpp(
                    "Hello", False, None, 60, None, 0.0, None)
            elif self.api_format == "openai":
                raw = self._generate_openai(
                    "Hello", False, None, 60, None, 0.0)
            else:
                raw = self._generate_koboldcpp(
                    "Hello", False, None, 60, None, 0.0, None)
            lower = (raw or "").lower()
            self._is_thinking_model = ("<think" in lower
                                       or "</think" in lower)
            log.info(f"[LLM] Thinking-model auto-detect: "
                     f"{self._is_thinking_model}")
        except Exception as e:
            log.warning(f"[LLM] Thinking-model probe failed ({e}); "
                      f"assuming non-thinking.")
            self._is_thinking_model = False

    @property
    def is_thinking_model(self) -> bool:
        """True if the loaded model emits <think>/<thinking> tags.

        Lazily probed on first access. Cached afterward.
        """
        self._ensure_thinking_detected()
        return bool(self._is_thinking_model)

    def generate(self, prompt, use_vision=False, screenshot_b64=None,
                 images_b64=None, max_length=None, system_prompt=None,
                 temperature=None, rep_pen=None,
                 detect_degeneration=True,
                 no_think=False, think=False):
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
            rep_pen: Override repetition penalty for this call.
            detect_degeneration: If True (default), scan output for
                n-gram repetition and run-on sentences and truncate if
                detected.  Callers that expect structured output (JSON
                action plans, tool-call JSON, etc.) should pass False —
                truncating JSON mid-string breaks the parser, and
                legitimate creative content can trip the run-on detector.
            no_think: Suppress reasoning on thinking models by prepending
                /no_think to the system prompt. Fast-path calls (routing,
                classification, short JSON) should pass True.  No-op on
                non-thinking models.
            think: Force reasoning on thinking models by prepending
                /think to the system prompt. max_length is multiplied by
                thinking_boost (default 2.5) to leave room for reasoning
                tokens.  No-op on non-thinking models.
        """
        # Normalise: merge legacy screenshot_b64 into images_b64 list.
        effective_images = list(images_b64) if images_b64 else []
        if screenshot_b64 and screenshot_b64 not in effective_images:
            effective_images.insert(0, screenshot_b64)
        effective_images = effective_images or None

        # Thinking-model directives: only affect behaviour when the loaded
        # model is a thinking model. Non-thinking models would emit the
        # literal "/no_think" as part of their reply, so we must gate this.
        # NOTE: Qwen 3+ respects /no_think and /think most reliably when
        # the directive sits at the END of the user prompt, NOT alone in
        # the system prompt. Observed failure: a system prompt of just
        # "/no_think" with no following content is ignored and the model
        # thinks anyway.
        effective_prompt = prompt
        effective_max_length = max_length
        if (no_think or think) and self.is_thinking_model:
            directive = "/no_think" if no_think else "/think"
            effective_prompt = f"{prompt}\n\n{directive}"
            if think and max_length:
                effective_max_length = int(max_length * self.thinking_boost)

        if self.api_format == "llamacpp":
            raw = self._generate_llamacpp(
                effective_prompt, use_vision, effective_images,
                effective_max_length, system_prompt, temperature, rep_pen)
        elif self.api_format == "openai":
            raw = self._generate_openai(
                effective_prompt, use_vision, effective_images,
                effective_max_length, system_prompt, temperature)
        else:
            raw = self._generate_koboldcpp(
                effective_prompt, use_vision, effective_images,
                effective_max_length, system_prompt, temperature, rep_pen)

        # Strip <think>/<thinking> reasoning blocks BEFORE any further
        # processing. Qwen 3+ and similar models wrap their chain-of-
        # thought in these tags. Leaving them in breaks JSON parsers,
        # argument extractors ("location = '<think>'"), and UIs.
        raw = _strip_think_tags(raw)

        if detect_degeneration:
            return _truncate_degeneration(raw)
        return raw

    # ── KoboldCpp Backend ─────────────────────────────────────

    def _generate_koboldcpp(self, prompt, use_vision=False, images_b64=None,
                            max_length=None, system_prompt=None, temperature=None,
                            rep_pen=None):
        """KoboldCpp native API: POST /api/v1/generate."""
        effective_max_length = max_length or self.max_length
        effective_temperature = (temperature if temperature is not None
                                 else self.temperature)
        effective_rep_pen = rep_pen if rep_pen is not None else self.rep_pen

        formatted_prompt = self._build_chatml_prompt(
            prompt, system_prompt, use_vision)

        payload = {
            "prompt": formatted_prompt,
            "max_length": effective_max_length,
            "temperature": effective_temperature,
            "top_p": self.top_p,
            "rep_pen": effective_rep_pen,
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
                    raise LLMError("Unexpected response format")
            else:
                raise LLMError(f"Status {response.status_code}",
                               status_code=response.status_code)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(str(e)) from e

    # ── llama.cpp Backend ─────────────────────────────────────

    def _generate_llamacpp(self, prompt, use_vision=False, images_b64=None,
                           max_length=None, system_prompt=None, temperature=None,
                           rep_pen=None):
        """llama.cpp server API: POST /completion."""
        # If we have a server manager reference, check readiness before sending.
        # Returns a friendly message instead of hanging/503 while model loads.
        if self.server_manager is not None:
            status = self.server_manager.status
            if status == "starting":
                return "I'm still loading the language model. Please try again in a moment."
            if status == "error":
                raise LLMError("LLM server failed to start. Check the LLM Server settings.")
            if status == "stopped":
                raise LLMError("LLM server is not running.")

        effective_max_length = max_length or self.max_length
        effective_temperature = (temperature if temperature is not None
                                 else self.temperature)
        effective_rep_pen = rep_pen if rep_pen is not None else self.rep_pen

        formatted_prompt = self._build_chatml_prompt(
            prompt, system_prompt, use_vision)

        payload = {
            "prompt": formatted_prompt,
            "n_predict": effective_max_length,
            "temperature": effective_temperature,
            "top_p": self.top_p,
            "repeat_penalty": effective_rep_pen,
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
                text = result.get("text", "").strip()
                if text:
                    return text
                raise LLMError("Unexpected response format")
            else:
                raise LLMError(f"Status {response.status_code}",
                               status_code=response.status_code)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(str(e)) from e

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
                raise LLMError("No choices in response")
            else:
                raise LLMError(f"Status {response.status_code}",
                               status_code=response.status_code)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(str(e)) from e

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
        log.debug(f"[LLM] Prompt size: {char_count} chars (~{approx_tokens} tokens)")

        return formatted
