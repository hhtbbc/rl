"""Core functionality tests for rl_course package."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """Test all core modules import correctly."""
    from rl_course import set_seed, get_device

    assert callable(set_seed)
    assert callable(get_device)


def test_seeding():
    """Test seed setting produces deterministic results."""
    from rl_course.utils.seeding import set_seed
    import numpy as np

    set_seed(42)
    a = np.random.randn(5)
    set_seed(42)
    b = np.random.randn(5)
    assert np.allclose(a, b)


def test_device():
    """Test device detection."""
    from rl_course.utils.seeding import get_device
    import torch

    device = get_device()
    assert isinstance(device, torch.device)

    device_cpu = get_device(force_cpu=True)
    assert device_cpu == torch.device("cpu")


def test_config():
    """Test Config dataclass."""
    from rl_course.utils.config import Config

    cfg = Config(seed=123, gamma=0.95, fast_mode=True)
    assert cfg.seed == 123
    assert cfg.gamma == 0.95
    assert cfg.fast_mode


def test_metric_tracker():
    """Test MetricTracker."""
    from rl_course.utils.logging import MetricTracker

    tracker = MetricTracker()
    tracker.add("reward", 100)
    tracker.add("reward", 200)
    tracker.add("reward", 300)

    assert len(tracker) == 1
    assert tracker.recent_mean("reward", window=2) == 250.0
    assert tracker.max("reward") == 300.0


def test_gridworld():
    """Test GridWorld environment."""
    from rl_course.envs.grid_world import GridWorld

    gw = GridWorld(width=4, height=4, start_pos=(0, 0), goal_pos=(3, 3))
    assert gw.n_states == 16
    assert gw.n_actions == 4

    state = gw.reset()
    assert state == 0  # Start at (0,0)

    # Test step
    next_state, reward, done, info = gw.step(1)  # Right
    assert next_state == 1
    assert reward == -1.0
    assert not done

    # Test transition matrix
    P = gw.get_transition_matrix()
    assert P.shape == (16, 4, 16)
    # Each (s,a) should sum to 1
    for s in range(16):
        for a in range(4):
            assert abs(P[s, a].sum() - 1.0) < 1e-6

    # Test render
    ansi = gw.render(mode="ansi")
    assert "A" in ansi
    assert "G" in ansi


def test_bandit():
    """Test MultiArmedBandit."""
    from rl_course.envs.bandit import MultiArmedBandit

    bandit = MultiArmedBandit(k=5, seed=42)
    assert bandit.k == 5
    assert 0 <= bandit.optimal_action < 5

    reward = bandit.pull(0)
    assert reward in (0.0, 1.0)  # Bernoulli

    bandit2 = MultiArmedBandit(k=3, reward_type="gaussian", seed=42)
    reward2 = bandit2.pull(0)
    assert isinstance(reward2, float)


def test_mlp_network():
    """Test MLP networks."""
    import torch
    from rl_course.networks.mlp import MLP, ValueNetwork, QNetwork, PolicyNetwork, ActorCriticNetwork

    batch_size = 8
    x = torch.randn(batch_size, 4)

    # MLP
    mlp = MLP(4, [32, 32], 2)
    out = mlp(x)
    assert out.shape == (batch_size, 2)

    # Value Network
    vnet = ValueNetwork(4)
    v = vnet(x)
    assert v.shape == (batch_size, 1)

    # Q Network
    qnet = QNetwork(4, 3)
    q = qnet(x)
    assert q.shape == (batch_size, 3)

    # Policy Network
    pnet = PolicyNetwork(4, 3)
    probs = pnet(x)
    assert probs.shape == (batch_size, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    # Actor-Critic
    ac = ActorCriticNetwork(4, 3, [32, 32])
    logits, value = ac(x)
    assert logits.shape == (batch_size, 3)
    assert value.shape == (batch_size, 1)

    # Test get_action
    action, log_prob, v = ac.get_action(x)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert v.shape == (batch_size,)


def test_replay_buffer():
    """Test ReplayBuffer."""
    import numpy as np
    from rl_course.buffers.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=10, state_dim=4)

    for i in range(15):
        buf.push(np.ones(4) * i, 0, 1.0, np.zeros(4), False)

    assert len(buf) == 10  # Max capacity

    states, actions, rewards, next_states, dones = buf.sample(3)
    assert states.shape == (3, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 4)
    assert dones.shape == (3,)


def test_rollout_buffer():
    """Test RolloutBuffer."""
    import numpy as np
    from rl_course.buffers.rollout_buffer import RolloutBuffer

    buf = RolloutBuffer(buffer_size=16, state_dim=4, gamma=0.99, gae_lambda=0.95)

    for i in range(16):
        buf.add(np.ones(4) * i, 0, 1.0, i == 15, 0.5, 0.8)

    assert len(buf) == 16

    buf.compute_gae(last_value=0.0)
    assert buf.returns.shape == (16,)
    assert buf.advantages.shape == (16,)

    batches = buf.get_minibatches(4, shuffle=False)
    assert len(batches) == 4  # 16/4 = 4 batches


def test_tabular_agents():
    """Test tabular RL agents."""
    import numpy as np
    from rl_course.agents.tabular import (
        TabularSARSAAgent,
        TabularQLearningAgent,
        TabularExpectedSARSAAgent,
    )

    # SARSA
    sarsa = TabularSARSAAgent(n_states=2, n_actions=2, alpha=0.1, seed=42)
    metrics = sarsa.update(0, 0, 1.0, 1, 1, False)
    assert "td_error" in metrics

    # Q-Learning
    ql = TabularQLearningAgent(n_states=2, n_actions=2, alpha=0.1, seed=42)
    metrics = ql.update(0, 0, 1.0, 1, False)
    assert "td_error" in metrics

    # Expected SARSA
    esarsa = TabularExpectedSARSAAgent(n_states=2, n_actions=2, alpha=0.1, seed=42)
    metrics = esarsa.update(0, 0, 1.0, 1, False)
    assert "td_error" in metrics


def test_all_agent_imports():
    """Test all agents can be imported."""
    from rl_course.agents import (
        BaseAgent,
        TabularMCAgent,
        TabularSARSAAgent,
        TabularQLearningAgent,
        DQNAgent,
        DoubleDQNAgent,
        REINFORCEAgent,
        REINFORCEWithBaselineAgent,
        A2CAgent,
        PPOAgent,
    )
    # Just verify imports work
    assert BaseAgent is not None
    assert PPOAgent is not None
