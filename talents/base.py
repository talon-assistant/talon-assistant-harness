import re
from abc import ABC, abstractmethod


class BaseTalent(ABC):
    """Base class for all Talon talents.

    Subclasses must define:
        name: str         — unique talent identifier
        description: str  — what this talent does (used by the LLM router)
        examples: list    — natural-language example commands (primary routing)

    And implement:
        execute(command, context) — perform the command

    Optional:
        keywords: list    — fallback trigger words (degraded mode only)
        priority: int     — display ordering in the sidebar
        routing_available — override to False when backend is unavailable
        get_config_schema() / update_config() / initialize()
    """

    name: str = ""
    description: str = ""
    keywords: list[str] = []
    examples: list[str] = []  # Natural-language example commands for LLM routing
    priority: int = 50

    def __init__(self):
        self._enabled = True
        self._config = {}  # Per-talent config from talents.json

    @property
    def routing_available(self) -> bool:
        """Whether this talent should appear in the LLM routing roster.

        Override to return False when the talent's backend is unavailable
        (e.g., Hue bridge disconnected). The router will exclude this
        talent from the LLM prompt entirely.
        """
        return True

    def can_handle(self, command: str) -> bool:
        """Keyword fallback for degraded mode (LLM unavailable).

        No longer the primary routing mechanism — the LLM router handles
        all intent classification.  Kept for backward compatibility and
        as an emergency fallback when the LLM is unreachable.
        """
        return self.keyword_match(command)

    @abstractmethod
    def execute(self, command: str, context: dict) -> dict:
        """Execute the command.

        Args:
            command: The user's command string (lowercased, punctuation stripped).
            context: Dict containing shared services:
                - 'llm': LLMClient instance
                - 'memory': MemorySystem instance
                - 'vision': VisionSystem instance
                - 'voice': VoiceSystem instance
                - 'config': Full settings dict
                - 'memory_context': str - pre-fetched relevant memory context
                - 'speak_response': bool - whether to use TTS

        Returns:
            dict with keys:
                - 'success': bool
                - 'response': str - text response to user
                - 'actions_taken': list[dict] - actions performed (for logging)
                - 'spoken': bool - whether the talent already spoke the response
        """
        pass

    def get_config_schema(self) -> dict:
        """Return config schema for this talent's configurable fields.

        Override to expose talent-specific settings in the GUI config dialog.

        Returns:
            dict with a 'fields' list. Each field dict has:
                - key (str): config key name
                - label (str): human-readable label for the form
                - type (str): one of 'string', 'password', 'int', 'float',
                              'bool', 'choice', 'list'
                - default: default value
                - min (int/float, optional): minimum for int/float
                - max (int/float, optional): maximum for int/float
                - step (float, optional): step increment for float
                - choices (list[str], optional): options for 'choice' type

            Return {} (empty dict) if no config is needed.

        Example:
            return {
                "fields": [
                    {"key": "max_results", "label": "Max Results",
                     "type": "int", "default": 5, "min": 1, "max": 20},
                    {"key": "api_key", "label": "API Key",
                     "type": "password", "default": ""},
                    {"key": "unit", "label": "Unit",
                     "type": "choice", "default": "metric",
                     "choices": ["metric", "imperial"]},
                ]
            }
        """
        return {}

    @property
    def talent_config(self) -> dict:
        """Return this talent's per-talent config dict."""
        return self._config

    def update_config(self, config: dict) -> None:
        """Update this talent's per-talent config at runtime.

        Called by TalentConfigDialog when the user saves changes.
        Override to handle hot-swap logic (e.g., reconnect to a bridge,
        change API endpoints, etc.).
        """
        self._config = config

    def initialize(self, config: dict) -> None:
        """Called after construction with the full settings dict.
        Override to perform setup that needs config values.
        """
        pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def keyword_match(self, command: str) -> bool:
        """Helper: check if any of this talent's keywords appear in the command.

        Uses word-boundary matching so 'note' doesn't match inside 'notepad'.
        Multi-word keywords (e.g. 'save a note') use simple substring match
        since they're specific enough not to cause false positives.
        """
        cmd_lower = command.lower()
        for kw in self.keywords:
            if " " in kw:
                # Multi-word keyword: substring match is fine
                if kw in cmd_lower:
                    return True
            else:
                # Single-word keyword: require word boundaries
                if re.search(rf'\b{re.escape(kw)}\b', cmd_lower):
                    return True
        return False

    def _extract_arg(
        self,
        llm,
        command: str,
        what: str,
        *,
        options: list | None = None,
        max_length: int = 20,
        temperature: float = 0.0,
        fallback: str | None = None,
    ) -> str | None:
        """Extract a single named value from a command string using the LLM.

        Standardised Pattern E extraction: one tight, deterministic LLM call
        to pull a single value out of natural language.  Suitable for any
        talent that needs to extract a simple scalar (location, colour, name,
        duration expressed as a single word/phrase, etc.).

        For complex structured extraction (JSON schemas, multi-field parsing,
        time expressions with system-prompt injection) keep the talent's own
        custom LLM call — this helper is not a replacement for those.

        Args:
            llm:         LLMClient instance (from ``context['llm']``).
            command:     The raw user command string.
            what:        Natural-language name of the thing to extract,
                         e.g. ``"location"``, ``"colour"``, ``"device name"``.
            options:     Optional list of allowed values.  When provided the
                         prompt instructs the model to pick from this list,
                         which reduces hallucination for enum-style args.
            max_length:  Max tokens to generate (default 20 — enough for any
                         place name or single-word value).
            temperature: Sampling temperature (default 0.0 — deterministic).
            fallback:    Value to return when the model replies ``"NONE"`` or
                         returns an empty string.  ``None`` by default.

        Returns:
            Extracted string, or ``fallback`` (default ``None``) if the model
            reports no value was found.

        Example::

            location = self._extract_arg(llm, command, "location") \\
                       or self._config.get("default_location", "")

            colour = self._extract_arg(
                llm, command, "colour",
                options=["red", "green", "blue", "white"],
                fallback="white",
            )
        """
        prompt = (
            f"Extract the {what} from this command.\n"
            f"Return only the {what}, nothing else.\n"
        )
        if options:
            prompt += f"Must be one of: {', '.join(options)}.\n"
        prompt += (
            f"If no {what} is mentioned, return the single word: NONE\n\n"
            f"Command: {command}"
        )
        try:
            raw = llm.generate(prompt, max_length=max_length, temperature=temperature)
            result = raw.strip().strip('"').strip("'")
            if not result or result.upper() == "NONE":
                return fallback
            return result
        except Exception:
            return fallback
