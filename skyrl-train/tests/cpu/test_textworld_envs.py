"""Tests for the TextWorld environments: action parsing, memory documents and the modality payload."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import DictConfig

from skyrl_gym.envs.textworld.env import TextWorldEnv
from skyrl_gym.envs.textworld.fast_env import COMMAND_HINT, FastTextWorldEnv
from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator
from skyrl_gym.envs.textworld.memory import TrajectoryMemory, parse_action
from skyrl_train.modalities.batching import build_modality_batches
from skyrl_train.modalities.types import SampleModalityData

PROMPT = [
    {"role": "system", "content": "You play TextWorld. Answer with [ACTION: <command>]."},
    {"role": "user", "content": "What is your next action?"},
]


@pytest.fixture(scope="module")
def game(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, list[str]]:
    """Compile a small TextWorld game and return its ``.z8`` path and walkthrough."""
    if shutil.which("tw-make") is None:
        pytest.skip("tw-make is not available")
    path = tmp_path_factory.mktemp("textworld") / "game.z8"
    subprocess.run(
        ["tw-make", "custom", "--world-size", "5", "--nb-objects", "6", "--quest-length", "4", "--seed", "1234"]
        + ["--output", str(path)],
        check=True,
        capture_output=True,
    )
    spec = json.loads(path.with_suffix(".json").read_text())
    walkthrough = [command for quest in spec["quests"] for command in quest["commands"]]
    return str(path), walkthrough


def _encoder_payloads(modalities: SampleModalityData) -> list:
    """Return what the memory encoder would receive for one sample."""
    batches = build_modality_batches([modalities.clone()])
    return [(occurrence.payload, occurrence.reserved_tokens) for b in batches.values() for occurrence in b.occurrences]


@pytest.mark.parametrize(
    "response, command",
    [
        ("Let me think.\n[ACTION: open fridge]", "open fridge"),
        ("[action:  go north ] trailing", "go north"),
        ("I will take the key\nbecause it helps", "take the key"),
        ("<think>plan</think>\nAction: go west", "go west"),
        ("go east ☃", "go east"),
        ("", ""),
    ],
)
def test_parse_action(response: str, command: str) -> None:
    assert parse_action(response) == command


def test_trajectory_memory_windows_documents() -> None:
    memory = TrajectoryMemory(turns_per_document=2, max_documents=2, max_observation_chars=5)
    for turn in range(1, 6):
        memory.add_turn(turn=turn, observation="abcdefgh", action=f"act{turn}", reward=float(turn == 5), score=0)
    assert len(memory.documents) == 2
    assert memory.documents[-1].startswith("Turns 3-4 | Reward: 0.0 | Score Change: 0")
    assert "Obs: abcde\n" in memory.documents[-1]

    memory.flush()
    assert len(memory.documents) == 2
    assert memory.documents[-1].endswith("Act: act5 -> Reward: 1.0")


@pytest.mark.parametrize("env_class", [TextWorldEnv, FastTextWorldEnv])
def test_walkthrough_wins_and_publishes_documents(env_class: type, game: tuple[str, list[str]]) -> None:
    game_file, walkthrough = game
    extras = {"game_file": game_file, "modalities": SampleModalityData(), "max_turns": 50}
    env = env_class(DictConfig({}), extras)

    chat, _ = env.init([dict(message) for message in PROMPT])
    assert chat[-1]["role"] == "user" and chat[-1]["content"]
    assert chat[0]["content"].endswith(COMMAND_HINT) == (env_class is FastTextWorldEnv)
    assert _encoder_payloads(extras["modalities"]) == []

    total_reward = 0.0
    for turn, command in enumerate(walkthrough, start=1):
        output = env.step(f"[ACTION: {command}]")
        total_reward += output["reward"]
        payloads = _encoder_payloads(extras["modalities"])
        expected_docs = len(env._memory.documents)
        assert (len(payloads) == 1) == (expected_docs > 0)
        if payloads:
            documents, reserved = payloads[0]
            assert reserved == 8 and documents == env._memory.documents
        if env_class is TextWorldEnv:
            assert expected_docs == turn
    env.close()

    assert output["done"] and output["observations"] == []
    assert env.get_metrics()["won"] and total_reward == env.get_metrics()["score"] > 0


def test_disabled_memory_publishes_nothing(game: tuple[str, list[str]]) -> None:
    game_file, walkthrough = game
    extras = {"game_file": game_file, "modalities": SampleModalityData()}
    env = TextWorldEnv(DictConfig({"enable_memory": False}), extras)
    env.init([dict(message) for message in PROMPT])
    for command in walkthrough[:2]:
        env.step(f"[ACTION: {command}]")
    env.close()
    assert extras["modalities"].plans == {} and extras["modalities"].payloads == {}


def test_fast_simulator_matches_inform7(game: tuple[str, list[str]]) -> None:
    game_file, walkthrough = game
    reference = TextWorldEnv(DictConfig({}), {"game_file": game_file})
    reference._reset_game()
    simulator = FastTextWorldSimulator(str(Path(game_file).with_suffix(".json")))
    simulator.reset()
    for command in walkthrough:
        _, reward, done, score, won = reference._step_game(command)
        _, sim_reward, sim_done, info = simulator.step(command)
        assert (sim_reward, sim_done, info["score"], info["won"]) == (reward, done, score, won)
    reference.close()
