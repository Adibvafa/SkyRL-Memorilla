"""TextWorld environment backed by the Inform7 engine (``.z8`` games)."""

from __future__ import annotations

import threading
from typing import Any

import textworld
from omegaconf import DictConfig

from skyrl_gym.envs.textworld.base import TextWorldMemoryEnv

_START_LOCK = threading.Lock()


class TextWorldEnv(TextWorldMemoryEnv):
    """Plays a compiled TextWorld game through ``textworld.start``.

    Defaults: one document per completed turn, up to 60 documents, observations in documents cut to 200 characters.
    """

    DEFAULT_MEMORY_WINDOW = 1
    DEFAULT_MAX_MEMORY_DOCS = 60
    DEFAULT_MAX_OBSERVATION_CHARS = 200

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None) -> None:
        """Create the environment; the game process starts on the first reset.

        Args:
            env_config: Environment section of the SkyRL config for this env class.
            extras: Per-sample data; must contain ``game_file``.
        """
        super().__init__(env_config, extras)
        self._env: textworld.Environment | None = None

    def _reset_game(self) -> str:
        """Start the game process if needed and reset it.

        Returns:
            The opening observation.
        """
        if self._env is None:
            with _START_LOCK:  # textworld's grammar parser is not thread-safe while a game starts
                self._env = textworld.start(self.game_file)
        return self._feedback(self._env.reset())

    def _step_game(self, command: str) -> tuple[str, float, bool, int, bool]:
        """Send one command to the game process.

        Args:
            command: Parsed game command.

        Returns:
            ``(observation, reward, done, score, won)`` after the command.
        """
        game_state, reward, done = self._env.step(command)
        score = int(getattr(game_state, "score", 0) or 0)
        won = bool(getattr(game_state, "won", False))
        return self._feedback(game_state), float(reward), bool(done), score, won

    def close(self) -> None:
        """Stop the game process."""
        if self._env is not None:
            self._env.close()
        self._env = None

    @staticmethod
    def _feedback(game_state: Any) -> str:
        """Return the game's text output as a string.

        Args:
            game_state: State returned by the TextWorld environment.

        Returns:
            The feedback text.
        """
        feedback = game_state.feedback
        if isinstance(feedback, bytes):
            return feedback.decode("utf-8", errors="replace")
        return str(feedback)
