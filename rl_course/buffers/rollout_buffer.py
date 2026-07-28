"""
Rollout 缓冲区

用于 A2C、PPO 等 on-policy 算法的轨迹存储。
支持 GAE 优势估计和多轮 minibatch 采样。

关键设计:
- 每条 transition 存储 next_value (= V(s_{t+1})，在 add() 时由调用者提供)
- 这避免了用 values[t+1] 间接推断下一状态价值，
  确保截断 (truncated) 后重置到新 episode 时，GAE 仍使用截断前最终状态的价值
- terminated 控制是否 bootstrap (TD target 中是否包含未来价值)
- dones 控制 GAE 累积是否跨 episode 边界重置
"""

from typing import Tuple, Optional, List
import numpy as np
import torch


class RolloutBuffer:
    """固定长度的轨迹存储缓冲区。"""

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
        self.dones = np.zeros(buffer_size, dtype=np.float32)       # episode 边界
        self.terminated = np.zeros(buffer_size, dtype=np.float32)  # 真正终止
        self.next_values = np.zeros(buffer_size, dtype=np.float32) # V(s_{t+1})
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)      # V(s_t)

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
        next_value: float,
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
            value: Critic 估计的当前状态价值 V(s_t)
            next_value: Critic 估计的下一状态价值 V(s_{t+1})
                        对于 terminated=True: next_value=0.0 (未来价值为零)
                        对于 truncated=True: next_value=V(final_obs) (截断前最终状态)
                        对于正常步: next_value=V(next_state)
                        此值由调用者在 env.step() 之后、env.reset() 之前计算
            terminated: 是否为真正的环境终止 (vs 时间截断)
        """
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.terminated[idx] = float(terminated)
        self.next_values[idx] = next_value
        self.log_probs[idx] = log_prob
        self.values[idx] = value

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_gae(self) -> None:
        """
        使用 GAE 计算 advantages 和 returns。

        每条 transition 已存储:
          - values[t]:   V(s_t)
          - next_values[t]: V(s_{t+1}) (由调用者在 add 时提供)
          - terminated[t]: 真正终止标志
          - dones[t]:      episode 边界

        公式:
          delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
          A_t = delta_t + gamma * lambda * A_{t+1} * (1 - done_t)
          G_t = A_t + V(s_t)

        terminated 和 truncated 的区别:
          - terminated=True: V(s_{t+1}) = 0, 不 bootstrap → next_value 应为 0
          - truncated=True (done=True, terminated=False):
            V(s_{t+1}) = V(final_obs) → next_value 应为截断前最终状态的价值
          - done 控制 GAE 累积重置 (两种结束都重置)
        """
        n = self.size
        gae = 0.0

        for t in reversed(range(n)):
            # bootstrap_mask: 只有真正终止才不 bootstrap
            bootstrap_mask = 1.0 - self.terminated[t]

            # TD residual: 使用存储的 next_value (V(s_{t+1}))
            delta = (
                self.rewards[t]
                + self.gamma * self.next_values[t] * bootstrap_mask
                - self.values[t]
            )

            # GAE 递推: episode 边界处重置累积
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - self.dones[t])
            self.advantages[t] = gae

        # Returns = advantages + values
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_minibatches(
        self, batch_size: int, shuffle: bool = True
    ) -> List[Tuple[torch.Tensor, ...]]:
        """将缓冲区数据切分为多个 minibatch。"""
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
            advantages = torch.FloatTensor(self.advantages[batch_indices]).to(self.device)
            old_log_probs = torch.FloatTensor(self.log_probs[batch_indices]).to(self.device)
            old_values = torch.FloatTensor(self.values[batch_indices]).to(self.device)

            batches.append((states, actions, returns, advantages, old_log_probs, old_values))

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
