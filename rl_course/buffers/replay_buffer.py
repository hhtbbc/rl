"""
经验回放缓冲区（Replay Buffer）

用于 DQN 等 off-policy 算法。
"""

from typing import List, Tuple, Optional
import numpy as np
import torch


class ReplayBuffer:
    """
    固定容量的循环经验回放缓冲区。

    存储格式：每个 transition 为 (state, action, reward, next_state, done)。

    Args:
        capacity: 最大存储容量
        state_dim: 状态维度（用于张量化）
        device: 存储设备
    """

    def __init__(self, capacity: int = 10000, state_dim: int = 1, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)

        # 预分配 numpy 数组（比 list 更高效）
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # terminated: 真正的环境终止 (vs 时间截断)
        # 用于 TD target 中决定是否 bootstrap
        self.terminated = np.zeros(capacity, dtype=np.float32)

        self.pos = 0  # 当前写入位置
        self.size = 0  # 当前已存储数量

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        terminated: bool = False,
    ) -> None:
        """
        存入一个 transition。

        Args:
            state: 状态向量，shape (state_dim,)
            action: 动作索引
            reward: 奖励
            next_state: 下一状态向量，shape (state_dim,)
            done: episode 是否结束 (terminated or truncated)
            terminated: 是否为真正的环境终止 (vs 时间截断)
                         True -> TD target 不 bootstrap (未来价值为 0)
                         False + done=True -> 时间截断, bootstrap 仍然需要
        """
        idx = self.pos % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.terminated[idx] = float(terminated)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        均匀随机采样一个 minibatch。

        Args:
            batch_size: 批次大小

        Returns:
            (states, actions, rewards, next_states, dones, terminated) — torch.Tensor
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        terminated = torch.FloatTensor(self.terminated[indices]).to(self.device)

        return states, actions, rewards, next_states, dones, terminated

    def __len__(self) -> int:
        return self.size
