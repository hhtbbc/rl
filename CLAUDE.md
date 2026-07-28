# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A 28-day Reinforcement Learning crash course as 26 Jupyter notebooks + a reusable Python package (`rl_course/`). Teaches RL from MDP fundamentals to PPO implementation, with full mathematical derivations and from-scratch PyTorch code. Headless-server compatible (no `plt.show()`, no `render_mode="human"`).

## Common commands

```bash
# Environment setup
uv sync                                          # create venv, install all deps
uv sync --frozen                                 # CI: exact reproduction from lockfile
uv run python -m ipykernel install --user --name rl-course --display-name "Python (RL Course)"

# Jupyter
uv run jupyter lab --no-browser --ip=0.0.0.0     # start notebook server

# Testing
uv run pytest -q                                 # run all tests
uv run pytest tests/test_core.py -v              # run with verbose output

# Notebook validation (fast mode skips 5 long-training notebooks)
uv run python scripts/validate_notebooks.py --fast --timeout 120
uv run python scripts/validate_notebooks.py --notebook 04_mdp.ipynb  # single notebook

# Lockfile
uv lock                                          # regenerate uv.lock after dependency changes
```

## Critical API conventions

### terminated vs truncated

This is the most important convention in the codebase. Every `env.step()` call MUST distinguish these two signals:

- **`terminated`**: environment reached a natural terminal state (goal, death). Future value = 0 — do NOT bootstrap.
- **`truncated`**: environment hit a time/max-step limit. State still has valid future value — MUST bootstrap.
- **`done = terminated or truncated`**: controls only env reset, NOT value bootstrapping.

In GAE and TD targets, the bootstrap mask is `1.0 - float(terminated)`, NOT `1.0 - float(done)`. Getting this wrong produces silently incorrect value estimates in environments with time limits.

### GridWorld step() signature

`GridWorld.step(action)` returns a **5-tuple**: `(observation, reward, terminated, truncated, info)`. This is gymnasium-compatible. Callers must unpack all 5 values. `GridWorld.reset()` returns `(obs, info)` tuple.

### RolloutBuffer.add() requires next_value

`buffer.add(state, action, reward, done, log_prob, value, next_value, terminated=...)` — `next_value` (= V(s_{t+1})) must be computed by the caller BEFORE `env.reset()`. This ensures truncated episodes bootstrap from the correct final observation rather than the reset state.

### Tabular TD agents: terminated parameter

All tabular agents (SARSA, Q-Learning, Expected SARSA, Double Q) accept `terminated: bool = False` in their `update()` methods. The TD target uses `1.0 - float(terminated)` for bootstrap masking. Callers must explicitly pass `terminated` — do not rely on the default `False`.

## Architecture

### Package structure

- **`rl_course/agents/`**: Algorithm implementations. `BaseAgent` is the abstract base. Key agents: `PPOAgent` (uses `RolloutBuffer` + `ActorCriticNetwork`), `DQNAgent`/`DoubleDQNAgent` (uses `ReplayBuffer` + `QNetwork`), `A2CAgent` (single-env n-step AC), tabular agents (pure numpy).
- **`rl_course/buffers/`**: `RolloutBuffer` stores per-transition data including `next_value`; `compute_gae()` uses stored values (takes no arguments). `ReplayBuffer` is a circular buffer returning 6-tuples from `sample()`.
- **`rl_course/networks/`**: `ActorCriticNetwork` is a shared-backbone network with actor and critic heads. `get_action(state, deterministic)` returns `(action, log_prob, value)`.
- **`rl_course/envs/`**: `GridWorld` — goal state is absorbing in `get_transition_matrix()`. `MultiArmedBandit` — Bernoulli/Gaussian reward.
- **`rl_course/visualization/`**: All plotting uses `matplotlib.use('Agg')`. Videos via `imageio` from `rgb_array` frames. Never `plt.show()`.

### Notebook ↔ package relationship

Notebooks import from `rl_course` for reusable components (networks, buffers, agents) but write training loops inline for pedagogical transparency. Key algorithm notebooks (10 DQN, 12 REINFORCE, 14 AC, 15 A2C, 20 PPO) contain from-scratch implementations that may use `rl_course.networks` and `rl_course.buffers` but implement training logic directly.

### PPO API

`PPOAgent` uses a two-phase interaction pattern:
1. `action = agent.act(obs)` — stores state/action/log_prob/value internally
2. After `env.step()`: compute `next_value = critic(next_obs)` before reset
3. `agent.store(reward, done, terminated, next_value)` — completes the transition
4. After `n_steps` collected: `agent.update()` — GAE + multi-epoch SGD

`store()` requires all 4 arguments — there is no default for `terminated` (will raise TypeError if omitted).

### Notebook validation

`scripts/validate_notebooks.py` executes notebooks via `nbconvert --execute`. Fast mode (`--fast`) skips 5 long-training notebooks (10/12/14/15/20). For full validation, run without `--fast`.

## Design constraints

- **Headless server**: No GUI. All rendering via `rgb_array` → imageio GIF/MP4 or `%matplotlib inline` in notebooks.
- **CPU-first**: All code runs on CPU. GPU auto-detected but optional.
- **No stable-baselines3 or similar RL libraries**: All algorithms implemented from scratch in `rl_course/agents/`.
- **uv for package management**: Never use conda, pip, or manual venv.
- **Fixed seeds**: `set_seed(42)` at the top of every training script/notebook for reproducibility.
- **Chinese + English**: Notebook explanations in Chinese, code identifiers in English.
