from abc import ABC, abstractmethod


class BaseTalent(ABC):
    """Abstract base class for all Talon talents.

    Subclasses must define:
        name: str - unique talent identifier
        description: str - human-readable description
        keywords: list[str] - trigger words for command routing
        priority: int - routing priority (higher = checked first)

    And implement:
        can_handle(command) - whether this talent handles the command
        execute(command, context) - perform the command

    Optionally override:
        get_config_schema() - declare configurable fields for the GUI
        update_config(config) - handle runtime config changes
        initialize(config) - one-time setup with the full settings dict
    """

    name: str = ""
    description: str = ""
    keywords: list[str] = []
    examples: list[str] = []  # Natural-language example commands for LLM routing
    priority: int = 50

    def __init__(self):
        self._enabled = True
        self._config = {}  # Per-talent config from talents.json

    @abstractmethod
    def can_handle(self, command: str) -> bool:
        """Return True if this talent can handle the given command."""
        pass

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
        """Helper: check if any of this talent's keywords appear in the command."""
        return any(kw in command for kw in self.keywords)
