"""Shared game loop for the TextWorld environments: turns, rewards and the memory modality payload."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.textworld.memory import TrajectoryMemory, parse_action
from skyrl_train.dataset.modalities import ModalityPlaceholderPlan

DEFAULT_MAX_TURNS = 50
DEFAULT_ACTION = "look"


class TextWorldMemoryEnv(BaseTextEnv):
    """A TextWorld game whose completed turns are exposed to a memory modality.

    After every turn the environment publishes the current memory documents as the payload of the modality
    ``memory_modality_id`` in ``extras["modalities"]``, with ``max_placeholder_tokens`` reserved tokens. The generator
    turns that payload into memory vectors placed before the prompt. The reward of a turn is the game score after it, as
    TextWorld reports it.

    Config keys (``env_config``): ``max_turns``, ``memory_window`` (completed turns per document),
    ``max_memory_docs`` (documents kept, most recent first), ``max_observation_chars`` (observation length kept in a
    document, 0 for no limit), ``memory_modality_id``, ``placeholder_token``, ``max_placeholder_tokens`` and
    ``enable_memory`` (False plays the game without any memory payload).
    """

    DEFAULT_MEMORY_WINDOW = 1
    DEFAULT_MAX_MEMORY_DOCS = 60
    DEFAULT_MAX_OBSERVATION_CHARS = 200

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None) -> None:
        """Read the game file and memory settings.

        Args:
            env_config: Environment section of the SkyRL config for this env class.
            extras: Per-sample data; must contain ``game_file`` (top level or under ``extra_info``).

        Raises:
            ValueError: If no game file is given.
        """
        super().__init__()
        self.extras = extras if extras is not None else {}

        game_file = self.extras.get("game_file") or self.extras.get("extra_info", {}).get("game_file")
        if not game_file:
            raise ValueError(f"{type(self).__name__} requires `game_file` in extras.")
        self.game_file = str(game_file)

        self.max_turns = int(self.extras.get("max_turns") or env_config.get("max_turns", DEFAULT_MAX_TURNS))
        self.memory_window = int(env_config.get("memory_window", self.DEFAULT_MEMORY_WINDOW))
        self.max_memory_docs = int(env_config.get("max_memory_docs", self.DEFAULT_MAX_MEMORY_DOCS))
        self.max_observation_chars = int(env_config.get("max_observation_chars", self.DEFAULT_MAX_OBSERVATION_CHARS))
        self.modality_id = str(env_config.get("memory_modality_id", "memorilla"))
        self.placeholder_token = str(env_config.get("placeholder_token", "<|image_pad|>"))
        self.max_placeholder_tokens = int(env_config.get("max_placeholder_tokens", 8))
        self.enable_memory = bool(env_config.get("enable_memory", True))

        self._observation = ""
        self._score = 0
        self._won = False
        self._memory = self._new_memory()

    @abstractmethod
    def _reset_game(self) -> str:
        """Start a new episode.

        Returns:
            The opening observation.
        """

    @abstractmethod
    def _step_game(self, command: str) -> tuple[str, float, bool, int, bool]:
        """Send one command to the game.

        Args:
            command: Parsed game command.

        Returns:
            ``(observation, reward, done, score, won)`` after the command.
        """

    def _system_prompt(self, content: str) -> str:
        """Return the system prompt used for this episode.

        Args:
            content: System prompt from the dataset row.

        Returns:
            The system prompt given to the agent.
        """
        return content

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        """Reset the game and return the opening conversation.

        Args:
            prompt: Chat messages from the dataset row; the first one is the system prompt.

        Returns:
            The prompt followed by the opening observation as a user message, and empty metadata.
        """
        self._observation = self._reset_game().strip()
        self.turns = 0
        self._score = 0
        self._won = False
        self._memory = self._new_memory()

        messages = [dict(message) for message in prompt]
        if messages:
            messages[0]["content"] = self._system_prompt(messages[0]["content"])
        self._publish_memory()
        return messages + [{"role": "user", "content": self._observation}], {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Play one turn.

        Args:
            action: Raw model response; the command is parsed from it and defaults to ``look``.

        Returns:
            The next observation (none once the episode ends), the turn reward, the done flag and metadata holding the
            command, the score, the win flag and the updated modality data.
        """
        self.turns += 1
        command = parse_action(action) or DEFAULT_ACTION
        observation, reward, done, score, won = self._step_game(command)

        self._memory.add_turn(
            turn=self.turns, observation=self._observation, action=command, reward=float(reward), score=int(score)
        )
        if done or self.turns >= self.max_turns:
            self._memory.flush()
        self._publish_memory()

        self._observation = observation.strip()
        self._score = int(score)
        self._won = bool(won)
        done = done or self.turns >= self.max_turns

        metadata: dict[str, Any] = {"action": command, "score": self._score, "won": self._won}
        if self.extras.get("modalities") is not None:
            metadata["modalities"] = self.extras["modalities"]
        return BaseTextEnvStepOutput(
            observations=[] if done else [{"role": "user", "content": self._observation}],
            reward=float(reward),
            done=done,
            metadata=metadata,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return episode statistics.

        Returns:
            Number of turns played, whether the game was won, and the final score.
        """
        return {"steps": self.turns, "won": self._won, "score": self._score}

    def _new_memory(self) -> TrajectoryMemory:
        """Create an empty trajectory memory with this environment's settings.

        Returns:
            The memory.
        """
        return TrajectoryMemory(
            turns_per_document=self.memory_window,
            max_documents=self.max_memory_docs,
            max_observation_chars=self.max_observation_chars,
        )

    def _publish_memory(self) -> None:
        """Write the current documents into the modality payload, or clear it when there are none yet."""
        modalities = self.extras.get("modalities")
        if not self.enable_memory or modalities is None or not hasattr(modalities, "payloads"):
            return

        documents = [document for document in self._memory.documents if document]
        if not documents:
            modalities.payloads.pop(self.modality_id, None)
            modalities.plans.pop(self.modality_id, None)
            return

        modalities.payloads[self.modality_id] = [documents]
        modalities.plans[self.modality_id] = ModalityPlaceholderPlan(
            modality_id=self.modality_id,
            placeholder_token=self.placeholder_token,
            occurrences=1,
            reserved_tokens=[self.max_placeholder_tokens],
            payload=[documents],
        )
