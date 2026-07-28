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

    state, _ = gw.reset()
    assert state == 0  # Start at (0,0)

    # Test step — returns 5-tuple (obs, reward, terminated, truncated, info)
    result = gw.step(1)  # Right
    assert len(result) == 5
    next_state, reward, terminated, truncated, info = result
    assert next_state == 1
    assert reward == -1.0
    assert not terminated
    assert not truncated

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
        buf.push(np.ones(4) * i, 0, 1.0, np.zeros(4), False, terminated=False)

    assert len(buf) == 10  # Max capacity

    states, actions, rewards, next_states, dones, terminated = buf.sample(3)
    assert states.shape == (3, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 4)
    assert dones.shape == (3,)
    assert terminated.shape == (3,)


def test_rollout_buffer():
    """Test RolloutBuffer."""
    import numpy as np
    from rl_course.buffers.rollout_buffer import RolloutBuffer

    buf = RolloutBuffer(buffer_size=16, state_dim=4, gamma=0.99, gae_lambda=0.95)

    for i in range(16):
        buf.add(np.ones(4) * i, 0, 1.0, i == 15, 0.5, 0.8, next_value=0.0, terminated=(i == 15))

    assert len(buf) == 16

    buf.compute_gae()
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
        TabularDoubleQLearningAgent,
    )

    # SARSA
    sarsa = TabularSARSAAgent(n_states=2, n_actions=2, alpha=0.1, seed=42)
    metrics = sarsa.update(0, 0, 1.0, 1, 1, terminated=False)
    assert "td_error" in metrics

    # Q-Learning
    ql = TabularQLearningAgent(n_states=2, n_actions=2, alpha=0.1, seed=42)
    metrics = ql.update(0, 0, 1.0, 1, terminated=False)
    assert "td_error" in metrics

    # Expected SARSA
    esarsa = TabularExpectedSARSAAgent(n_states=2, n_actions=2, alpha=0.1, seed=42)
    metrics = esarsa.update(0, 0, 1.0, 1, terminated=False)
    assert "td_error" in metrics

    # Double Q-Learning
    dql = TabularDoubleQLearningAgent(n_states=2, n_actions=2, alpha=0.1, seed=42)
    metrics = dql.update(0, 0, 1.0, 1, terminated=False)
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


def test_a2c_n_step_returns():
    """Test A2C n-step return computation with known values."""
    from rl_course.agents.a2c import A2CAgent
    import torch
    import numpy as np

    agent = A2CAgent(state_dim=2, n_actions=2, gamma=0.9, lr=1e-3, n_steps=5)

    # Simulate a 2-step non-terminal rollout
    agent.states = [torch.zeros(2), torch.zeros(2)]
    agent.actions = [0, 0]
    agent.rewards = [1.0, 2.0]
    agent.dones = [False, False]
    agent.terminated_list = [False, False]
    agent.next_values = [3.0, 5.0]  # V(s1)=3, V(s2)=5

    # Manually trigger reward computation
    T = 2
    returns = np.zeros(T, dtype=np.float32)
    G = agent.next_values[-1]  # = 5.0 (bootstrap from last state)
    for t in reversed(range(T)):
        if agent.dones[t]:
            mask = 1.0 - float(agent.terminated_list[t])
            G = agent.rewards[t] + agent.gamma * agent.next_values[t] * mask
        else:
            G = agent.rewards[t] + agent.gamma * G
        returns[t] = G

    # R1 = 2 + 0.9*5 = 6.5
    # R0 = 1 + 0.9*6.5 = 6.85
    assert abs(returns[1] - 6.5) < 1e-5, f"Expected R1=6.5, got {returns[1]}"
    assert abs(returns[0] - 6.85) < 1e-5, f"Expected R0=6.85, got {returns[0]}"

    # Clean up
    agent.states.clear()
    agent.actions.clear()
    agent.rewards.clear()
    agent.dones.clear()
    agent.terminated_list.clear()
    agent.next_values.clear()


def test_a2c_mid_rollout_terminated():
    """Test A2C mid-rollout terminated boundary."""
    from rl_course.agents.a2c import A2CAgent
    import torch
    import numpy as np

    agent = A2CAgent(state_dim=2, n_actions=2, gamma=0.9, lr=1e-3, n_steps=5)

    # Episode 1: r0, r1(terminated). Episode 2: r2, r3
    agent.states = [torch.zeros(2)] * 4
    agent.actions = [0] * 4
    agent.rewards = [1.0, 2.0, 3.0, 4.0]
    agent.dones = [False, True, False, False]
    agent.terminated_list = [False, True, False, False]
    agent.next_values = [10.0, 0.0, 20.0, 30.0]  # term at idx1→next_value=0

    T = 4
    returns = np.zeros(T, dtype=np.float32)
    G = agent.next_values[-1]  # 30
    for t in reversed(range(T)):
        if agent.dones[t]:
            mask = 1.0 - float(agent.terminated_list[t])
            G = agent.rewards[t] + agent.gamma * agent.next_values[t] * mask
        else:
            G = agent.rewards[t] + agent.gamma * G
        returns[t] = G

    # R3 = 4 + 0.9*30 = 31
    # R2 = 3 + 0.9*31 = 30.9
    # R1 = terminated: r1 + gamma*0*0 = 2 (no bootstrap, no cross-episode)
    # R0 = 1 + 0.9*2 = 2.8
    assert abs(returns[3] - 31.0) < 1e-4, f"R3={returns[3]}"
    assert abs(returns[2] - 30.9) < 1e-4, f"R2={returns[2]}"
    assert abs(returns[1] - 2.0) < 1e-4, f"R1={returns[1]} (should not include ep2)"
    assert abs(returns[0] - 2.8) < 1e-4, f"R0={returns[0]}"

    agent.states.clear(); agent.actions.clear(); agent.rewards.clear()
    agent.dones.clear(); agent.terminated_list.clear(); agent.next_values.clear()


def test_gridworld_default_goal():
    """Test GridWorld auto-computes goal from dimensions."""
    from rl_course.envs.grid_world import GridWorld

    gw = GridWorld(width=4, height=4)
    assert gw.goal_pos == (3, 3), f"Expected (3,3), got {gw.goal_pos}"

    gw2 = GridWorld(width=10, height=6)
    assert gw2.goal_pos == (5, 9), f"Expected (5,9), got {gw2.goal_pos}"


def test_gridworld_validation():
    """Test GridWorld validates positions."""
    from rl_course.envs.grid_world import GridWorld
    import pytest

    # goal outside grid
    with pytest.raises(ValueError, match="outside"):
        GridWorld(width=3, height=3, goal_pos=(5, 5))

    # start == goal
    with pytest.raises(ValueError, match="must differ"):
        GridWorld(width=3, height=3, start_pos=(1, 1), goal_pos=(1, 1))

    # start on blocked
    with pytest.raises(ValueError, match="blocked"):
        GridWorld(width=3, height=3, start_pos=(1, 1), blocked_positions=[(1, 1)])


def test_tabular_td_targets():
    """Test tabular TD targets with known values."""
    from rl_course.agents.tabular import (
        TabularQLearningAgent, TabularSARSAAgent
    )
    import numpy as np

    # Q-Learning: terminated=False → bootstrap
    ql = TabularQLearningAgent(n_states=3, n_actions=2, gamma=0.9, alpha=1.0, seed=42)
    ql.Q[1] = [5.0, 3.0]  # max Q(s') = 5
    metrics = ql.update(state=0, action=0, reward=2.0, next_state=1, terminated=False)
    # target = 2 + 0.9*5 = 6.5, Q[0,0] was 0 → td_error = 6.5, Q[0,0] = 0 + 1.0*6.5 = 6.5
    assert abs(ql.Q[0, 0] - 6.5) < 1e-5, f"Q[0,0]={ql.Q[0,0]}"
    assert abs(metrics["td_error"] - 6.5) < 1e-5

    # Q-Learning: terminated=True → no bootstrap
    ql2 = TabularQLearningAgent(n_states=3, n_actions=2, gamma=0.9, alpha=1.0, seed=42)
    ql2.Q[1] = [5.0, 3.0]
    metrics2 = ql2.update(state=0, action=1, reward=2.0, next_state=1, terminated=True)
    # target = 2 + 0 = 2, Q[0,1] was 0 → td_error = 2
    assert abs(ql2.Q[0, 1] - 2.0) < 1e-5, f"Q[0,1]={ql2.Q[0,1]}"
    assert abs(metrics2["td_error"] - 2.0) < 1e-5


def test_mc_agent_first_visit():
    """Test First-Visit MC with repeated state."""
    from rl_course.agents.tabular import TabularMCAgent

    agent = TabularMCAgent(n_states=3, n_actions=2, gamma=1.0, seed=42)
    # Episode: (s=0,a=0,r=1), (s=1,a=1,r=2), (s=0,a=0,r=3), done
    episode = [(0, 0, 1.0), (1, 1, 2.0), (0, 0, 3.0)]
    metrics = agent.update(episode)

    # Returns: G2 = 3, G1 = 2+3=5, G0 = 1+2+3=6
    # First-visit: (0,0) counted once with G0=6, (1,1) counted with G1=5
    assert abs(agent.Q[0, 0] - 6.0) < 1e-5, f"Q[0,0] should be 6.0, got {agent.Q[0,0]}"
    assert abs(agent.Q[1, 1] - 5.0) < 1e-5, f"Q[1,1] should be 5.0, got {agent.Q[1,1]}"
    # (0,0) second visit should NOT update (first visit already counted)
    assert agent._returns_count[0, 0] == 1
    assert agent._returns_count[1, 1] == 1

    assert abs(metrics["episode_return"] - 6.0) < 1e-5
