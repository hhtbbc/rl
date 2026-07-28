"""
表格式强化学习算法

包括：
- Monte Carlo (first-visit MC prediction, MC control with exploring starts)
- SARSA (on-policy TD control)
- Q-Learning (off-policy TD control)
- Expected SARSA
- Double Q-Learning

全部基于 numpy 数组实现，不依赖 PyTorch。
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from rl_course.agents.base import BaseAgent


class TabularMCAgent(BaseAgent):
    """
    First-Visit Monte Carlo 控制（Exploring Starts）。

    使用 exploring starts 保证所有状态-动作对被探索。
    适用于 episodic 环境。

    Args:
        n_states: 状态数量
        n_actions: 动作数量
        gamma: 折扣因子
        epsilon: ε-greedy 探索率（0 表示纯 exploiting starts）
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon

        # Q 表: shape (n_states, n_actions)，初始化为 0
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

        # 每状态-动作对的回报累积
        self._returns_sum = np.zeros((n_states, n_actions), dtype=np.float32)
        self._returns_count = np.zeros((n_states, n_actions), dtype=np.float32)

        # 确定性策略：π(s) = argmax_a Q(s, a)
        self.policy = np.zeros(n_states, dtype=np.int32)

    def act(self, obs: int, train: bool = True) -> int:
        """ε-greedy 动作选择"""
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[obs]))

    def update(self, episode: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """
        用一条完整 episode 更新 Q 值 (First-visit MC)。

        正确实现: 先反向计算所有时间步的回报 G_t,
        再正向遍历 episode, 只取每个 (s,a) 的首次出现。

        Args:
            episode: [(state, action, reward), ...] 列表

        Returns:
            字典形式的训练指标
        """
        T = len(episode)

        # Step 1: 反向计算所有 G_t
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = episode[t][2] + self.gamma * G
            returns[t] = G

        # Step 2: 正向筛选 First-visit
        visited = set()
        for t in range(T):
            state, action, _ = episode[t]
            if (state, action) not in visited:
                visited.add((state, action))
                self._returns_sum[state, action] += returns[t]
                self._returns_count[state, action] += 1
                self.Q[state, action] = (
                    self._returns_sum[state, action]
                    / self._returns_count[state, action]
                )

        # 更新策略
        self.policy = np.argmax(self.Q, axis=1)

        return {"episode_return": returns[0]}


class TabularSARSAAgent(BaseAgent):
    """
    SARSA (State-Action-Reward-State-Action) — on-policy TD control。

    更新公式:
        Q(S, A) ← Q(S, A) + α [R + γ Q(S', A') - Q(S, A)]

    Args:
        n_states: 状态数量
        n_actions: 动作数量
        gamma: 折扣因子
        alpha: 学习率
        epsilon: ε-greedy 探索率
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, obs: int, train: bool = True) -> int:
        """ε-greedy 动作选择"""
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[obs]))

    def update(
        self, state: int, action: int, reward: float,
        next_state: int, next_action: int,
        done: bool, terminated: bool = False,
    ) -> Dict[str, float]:
        """
        SARSA 单步更新。

        bootstrap_mask = 1 - terminated: 只有真正终止才不 bootstrap。
        时间截断 (done=True, terminated=False) 仍需从 next_state bootstrap。

        Returns:
            {"td_error": td_error}
        """
        current_q = self.Q[state, action]
        bootstrap = 1.0 - float(terminated)

        target = reward + self.gamma * self.Q[next_state, next_action] * bootstrap

        td_error = target - current_q
        self.Q[state, action] += self.alpha * td_error

        return {"td_error": float(td_error)}


class TabularQLearningAgent(BaseAgent):
    """
    Q-Learning — off-policy TD control。

    更新公式:
        Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) - Q(S, A)]

    关键区别：使用 max_a Q(S', a) 而非实际采取的 A'。
    这意味着 Q-Learning 学习的是最优策略，而行为策略可以是任意的。

    Args:
        n_states: 状态数量
        n_actions: 动作数量
        gamma: 折扣因子
        alpha: 学习率
        epsilon: 探索率
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, obs: int, train: bool = True) -> int:
        """ε-greedy (行为策略)"""
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[obs]))

    def update(
        self, state: int, action: int, reward: float,
        next_state: int, done: bool, terminated: bool = False,
    ) -> Dict[str, float]:
        """
        Q-Learning 单步更新。

        bootstrap_mask = 1 - terminated: 只有真正终止才不 bootstrap。

        Returns:
            {"td_error": td_error}
        """
        current_q = self.Q[state, action]
        bootstrap = 1.0 - float(terminated)

        target = reward + self.gamma * np.max(self.Q[next_state]) * bootstrap

        td_error = target - current_q
        self.Q[state, action] += self.alpha * td_error

        return {"td_error": float(td_error)}


class TabularExpectedSARSAAgent(BaseAgent):
    """
    Expected SARSA — 使用期望值代替采样值。

    更新公式:
        Q(S, A) ← Q(S, A) + α [R + γ Σ_a π(a|S') Q(S', a) - Q(S, A)]

    相比 SARSA 降低了方差（因为求期望而非采样 A'）。
    相比 Q-Learning 更稳定（考虑了探索策略的分布）。
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, obs: int, train: bool = True) -> int:
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[obs]))

    def _expected_value(self, state: int) -> float:
        """计算状态 s 下的期望 Q 值"""
        best_action = np.argmax(self.Q[state])
        n = self.n_actions

        # 每个动作的概率
        probs = np.ones(n) * (self.epsilon / n)
        probs[best_action] += 1 - self.epsilon

        return float(np.sum(probs * self.Q[state]))

    def update(
        self, state: int, action: int, reward: float,
        next_state: int, done: bool, terminated: bool = False,
    ) -> Dict[str, float]:
        current_q = self.Q[state, action]
        bootstrap = 1.0 - float(terminated)

        target = reward + self.gamma * self._expected_value(next_state) * bootstrap

        td_error = target - current_q
        self.Q[state, action] += self.alpha * td_error

        return {"td_error": float(td_error)}


class TabularDoubleQLearningAgent(BaseAgent):
    """
    Double Q-Learning — 解决 maximization bias。

    维护两个 Q 表 Q_A 和 Q_B，每次更新随机选一个：
    - 用 Q_A 选择最优动作，用 Q_B 评估
    - 或用 Q_B 选择最优动作，用 Q_A 评估

    更新公式（选取 Q_A 时）:
        A* = argmax_a Q_A(S', a)
        Q_A(S, A) ← Q_A(S, A) + α [R + γ Q_B(S', A*) - Q_A(S, A)]
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.QA = np.zeros((n_states, n_actions), dtype=np.float32)
        self.QB = np.zeros((n_states, n_actions), dtype=np.float32)

    @property
    def Q(self):
        """合并的 Q 表（用于动作选择）"""
        return (self.QA + self.QB) / 2.0

    def act(self, obs: int, train: bool = True) -> int:
        if train and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[obs]))

    def update(
        self, state: int, action: int, reward: float,
        next_state: int, done: bool, terminated: bool = False,
    ) -> Dict[str, float]:
        """Double Q-Learning 更新"""
        bootstrap = 1.0 - float(terminated)

        if np.random.rand() < 0.5:
            current_q = self.QA[state, action]
            best_action = np.argmax(self.QA[next_state])
            target = reward + self.gamma * self.QB[next_state, best_action] * bootstrap
            td_error = target - current_q
            self.QA[state, action] += self.alpha * td_error
        else:
            current_q = self.QB[state, action]
            best_action = np.argmax(self.QB[next_state])
            target = reward + self.gamma * self.QA[next_state, best_action] * bootstrap
            td_error = target - current_q
            self.QB[state, action] += self.alpha * td_error

        return {"td_error": float(td_error)}
