# TextWorld environments

Two multi-turn environments for [TextWorld](https://github.com/microsoft/TextWorld) games. Both expose the agent's
completed turns as documents for a memory modality, so a trainable memory module can condition the decoder on the
trajectory.

| Env id | Class | Game format | Engine |
|---|---|---|---|
| `textworld` | `TextWorldEnv` (`env.py`) | compiled `.z8` | Inform7 through `textworld.start` |
| `fast_textworld` | `FastTextWorldEnv` (`fast_env.py`) | `.json` game spec | `FastTextWorldSimulator` (`fast_sim.py`), in process |

`fast_textworld` needs no Inform7 compile, which makes it the practical choice for long games. Its score, reward, done
and win signals match the Inform7 engine on walkthroughs.

## Episode

- The dataset row supplies the chat prompt and `game_file`. `init` resets the game and appends the opening observation
  as a user message. `fast_textworld` also appends the command forms the parser accepts to the system prompt.
- Each model response is parsed for `[ACTION: <command>]` (with a first-line fallback); an empty command becomes `look`.
- The reward of a turn is the change in game score. The episode ends on a win, a loss or after `max_turns` turns.

## Memory documents

After every turn the environment writes its documents to `extras["modalities"]` under `memory_modality_id`, as one
occurrence of `max_placeholder_tokens` reserved tokens. A document covers `memory_window` completed turns:

```
Turns 3-3 | Reward: 0.0 | Score Change: 0

Turn 3:
Obs: -= Kitchen =- You arrive in a kitchen. ...
Act: open fridge
```

| Key | `textworld` | `fast_textworld` | Meaning |
|---|---|---|---|
| `memory_window` | 1 | 5 | completed turns per document |
| `max_memory_docs` | 60 | 4 | most recent documents kept |
| `max_observation_chars` | 200 | 0 | observation length kept per turn (0 keeps it whole) |
| `enable_memory` | true | true | false plays the game without a memory payload |

Defaults live in `skyrl_train/config/skyrl_gym_config/default.yaml` and can be overridden with
`environment.skyrl_gym.<env id>.<key>=<value>`.
