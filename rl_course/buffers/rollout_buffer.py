"""
Rollout 缓冲区

用于 A2C、PPO 等 on-policy 算法的轨迹存储。
支持 GAE 优势估计和多轮 minibatch 采样。
"""

from typing import Tuple, Optional, List
import numpy as np
import torch


class RolloutBuffer:
    """
    固定长度的轨迹存储缓冲区。

    用于在 on-policy 更新前收集完整的 rollout 数据。

    Args:
        buffer_size: 缓冲区大小（步数）
        state_dim: 状态维度
        gamma: 折扣因子
        gae_lambda: GAE λ 参数
        device: 存储设备
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        state_dim: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        # 预分配 numpy 数组
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        # dones: episode 边界 (terminated OR truncated) — 用于重置 GAE 累积
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        # terminated: 真正的环境终止 — 用于决定是否 bootstrap
        self.terminated = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)

        # GAE 计算结果
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        terminated: bool = False,
    ) -> None:
        """
        存入一步数据。

        Args:
            state: 状态向量，shape (state_dim,)
            action: 动作
            reward: 奖励
            done: episode 是否结束 (terminated or truncated)
            log_prob: 动作的 log probability
            value: Critic 估计的状态价值
            terminated: 是否为真正的环境终止 (vs 时间截断)
                         True → 不 bootstrap（未来价值为 0）
                         False + done=True → 时间截断，仍需 bootstrap
        """
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.terminated[idx] = float(terminated)
        self.log_probs[idx] = log_prob
        self.values[idx] = value

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float = 0.0) -> None:
        """
        使用 GAE 计算 advantages 和 returns。

        公式:
            δₜ = rₜ + γ * V(sₜ₊₁) * bootstrap_maskₜ - V(sₜ)
            Aₜ^GAE = Σ_{l=0}^{∞} (γλ)^l * δₜ₊ₗ
            Gₜ = Aₜ + V(sₜ)

        关键设计:
            - bootstrap_mask = 1.0 - terminatedₜ
              只有真正终止 (terminated) 才不 bootstrap。
              时间截断 (truncated) 时仍需用 V(s') 估计未来。
            - donesₜ 用于重置 GAE 累积 (无论终止还是截断，
              episode 边界后 GAE 从零开始)。

        Args:
            last_value: 最后状态的 Critic 估计
                        若 rollout 因 n_steps 截断，应传入 V(last_state)
                        若 episode 真正终止，传入 0.0
        """
        n = self.size
        gae = 0.0

        for t in reversed(range(n)):
            next_value = self.values[t + 1] if t + 1 < n else last_value

            # bootstrap_mask: 只有真正终止 (goal reached) 才不 bootstrap
            # 时间截断 (truncated) 时仍需 estimate 未来价值
            bootstrap_mask = 1.0 - self.terminated[t]

            # TD residual: δₜ
            delta = (
                self.rewards[t]
                + self.gamma * next_value * bootstrap_mask
                - self.values[t]
            )

            # GAE 递推: Aₜ = δₜ + γλ * Aₜ₊₁ * (1 - episode_boundaryₜ)
            # 在 episode 边界 (terminated or truncated) 处重置 GAE
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - self.dones[t])
            self.advantages[t] = gae

        # Returns = advantages + values
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_minibatches(
        self, batch_size: int, shuffle: bool = True
    ) -> List[Tuple[torch.Tensor, ...]]:
        """
        将缓冲区数据切分为多个 minibatch。

        Args:
            batch_size: minibatch 大小
            shuffle: 是否打乱

        Returns:
            list of (states, actions, returns, advantages, old_log_probs, old_values)
        """
        n = self.size
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            states = torch.FloatTensor(self.states[batch_indices]).to(self.device)
            actions = torch.LongTensor(self.actions[batch_indices]).to(self.device)
            returns = torch.FloatTensor(self.returns[batch_indices]).to(self.device)
            advantages = torch.FloatTensor(self.advantages[batch_indices]).to(
                self.device
            )
            old_log_probs = torch.FloatTensor(self.log_probs[batch_indices]).to(
                self.device
            )
            old_values = torch.FloatTensor(self.values[batch_indices]).to(self.device)

            batches.append(
                (states, actions, returns, advantages, old_log_probs, old_values)
            )

        return batches

    def reset(self) -> None:
        """重置缓冲区位置（不清除数据，覆盖写入）"""
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        """当前有效数据量"""
        return self.buffer_size if self.full else self.pos

    def __len__(self) -> int:
        return self.size
