"""TextWorld environment backed by the in-process simulator (``.json`` game specs)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.textworld.base import TextWorldMemoryEnv
from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator

COMMAND_HINT = """

Valid TextWorld commands (use exactly these forms — variations are rejected by the parser):
- `look` (describe room — NOT `look around` or `look at <room>`)
- `inventory` (list held items)
- `examine <object>` (NOT `examine <room>` or `inspect`)
- `take <object>` (NOT `pick up`, `grab`, `get`)
- `drop <object>`
- `open <object>` / `close <object>` (for doors and containers)
- `unlock <object> with <key>`
- `put <object> in <container>` / `put <object> on <surface>`
- `go <direction>` or just `<direction>` (north/south/east/west/up/down) — NOT `move`, `walk`, `head`
If a command is rejected (`You can't see any such thing.` or similar), DO NOT repeat it — try a different valid form."""


class FastTextWorldEnv(TextWorldMemoryEnv):
    """Plays a TextWorld game spec with ``FastTextWorldSimulator``, without an Inform7 process.

    Suited to long games, whose specs are generated without compiling them. The system prompt is extended with the
    command forms the parser accepts. Defaults: one document per five completed turns, the four most recent documents,
    full observations in documents.
    """

    DEFAULT_MEMORY_WINDOW = 5
    DEFAULT_MAX_MEMORY_DOCS = 4
    DEFAULT_MAX_OBSERVATION_CHARS = 0

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None) -> None:
        """Resolve the ``.json`` spec of the game.

        Args:
            env_config: Environment section of the SkyRL config for this env class.
            extras: Per-sample data; ``game_file`` may name the ``.json`` spec or the matching ``.z8`` game.
        """
        super().__init__(env_config, extras)
        if self.game_file.endswith(".json"):
            self.game_json = self.game_file
        elif self.game_file.endswith(".z8"):
            self.game_json = self.game_file[: -len(".z8")] + ".json"
        else:
            self.game_json = self.game_file + ".json"
        self._sim: FastTextWorldSimulator | None = None

    def _reset_game(self) -> str:
        """Load the game spec into a fresh simulator.

        Returns:
            The opening observation.
        """
        self._sim = FastTextWorldSimulator(self.game_json)
        observation, _ = self._sim.reset()
        return observation

    def _step_game(self, command: str) -> tuple[str, float, bool, int, bool]:
        """Apply one command in the simulator.

        Args:
            command: Parsed game command.

        Returns:
            ``(observation, reward, done, score, won)`` after the command.
        """
        observation, reward, done, info = self._sim.step(command)
        return observation, float(reward), bool(done), int(info.get("score", 0)), bool(info.get("won", False))

    def _system_prompt(self, content: str) -> str:
        """Append the accepted command forms to the system prompt.

        Args:
            content: System prompt from the dataset row.

        Returns:
            The extended system prompt.
        """
        return content.rstrip() + COMMAND_HINT

    def close(self) -> None:
        """Release the simulator."""
        self._sim = None
