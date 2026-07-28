"""
A2C (Advantage Actor-Critic) 算法实现

Advantage Actor-Critic 结合了策略梯度（Actor）和价值学习（Critic），
使用 n-step 回报来平衡偏差和方差。
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_course.agents.base import BaseAgent
from rl_course.networks.mlp import ActorCriticNetwork


def compute_n_step_returns(
    rewards: List[float],
    dones: List[bool],
    terminated_list: List[bool],
    next_values: List[float],
    gamma: float,
) -> np.ndarray:
    """
    计算 n-step 回报，正确处理 episode boundary。

    关键: done[t] 在 G 累积之前检查，防止跨 episode 污染。

    例如 rollout: [r0, r1, r2(done,term), r3, r4]
      反向遍历到 r2 时，当前 G 不应包含 r3, r4 的贡献。
      正确做法: 先检查 done[2]=True, 用 next_value[2] bootstrap, 再设 R2。

    Args:
        rewards: 奖励序列, length T
        dones: episode 边界 (terminated or truncated), length T
        terminated_list: 真正终止标志, length T
        next_values: V(s_{t+1}), 在 env.reset() 前计算, length T
        gamma: 折扣因子

    Returns:
        returns: shape (T,) numpy float32 array
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)

    # 初始化 G: rollout 末尾非 boundary → bootstrap from last next_value
    if dones[-1]:
        G = 0.0
    else:
        G = next_values[-1]

    for t in reversed(range(T)):
        if dones[t]:
            # Episode 边界: 回报仅包含本 transition
            # terminated=False (truncated): bootstrap from V(s_{t+1})
            # terminated=True: future value = 0
            bootstrap_mask = 1.0 - float(terminated_list[t])
            G = rewards[t] + gamma * next_values[t] * bootstrap_mask
        else:
            G = rewards[t] + gamma * G
        returns[t] = G

    return returns


class A2CAgent(BaseAgent):
    """
    Single-environment n-step Advantage Actor-Critic。

    注意: 这是单环境版本，用于教学。标准 A2C 使用多个同步环境并行采样。
    本实现演示 n-step return、advantage、entropy bonus 等核心概念。

    API:
        action = agent.act(obs)                          # 选择动作
        agent.store(reward, terminated, truncated,
                    next_value)                           # 存储环境反馈
        metrics = agent.update()                          # n-step 更新

    Episode boundary 处理:
        - 每一步存储 next_value = V(s_{t+1}) (在 env.reset() 之前计算)
        - 使用 compute_n_step_returns() 公共函数
        - done=True 处: G = r + γ·(1-terminated)·next_value (不跨 episode)
        - done=False 处: G = r + γ·G (正常累积)

    Loss = L_policy + L_value - entropy_coef * H(pi)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Optional[List[int]] = None,
        gamma: float = 0.99,
        lr: float = 1e-3,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.actor_critic = ActorCriticNetwork(
            state_dim, n_actions, hidden_dims
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Rollout 缓存
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []               # episode 边界 (terminated or truncated)
        self.terminated_list: List[bool] = []     # 真正终止
        self.next_values: List[float] = []         # V(s_{t+1}), 在 env.reset() 前计算

    def store(
        self, reward: float, terminated: bool, truncated: bool,
        next_value: float,
    ) -> None:
        """
        存储环境反馈。next_value 必须显式传入，无默认值以避免遗漏。

        调用顺序:
            1. act(obs)              — 获取动作
            2. env.step(action)       — 执行动作
            3. 计算 next_value = V(next_obs) (在 env.reset() 之前!)
            4. store(r, term, trunc, next_value) — 写入缓存
            5. if done: obs, _ = env.reset()

        Args:
            reward: 即时奖励
            terminated: 环境真正终止 (goal, death)
            truncated: 时间截断
            next_value: V(s_{t+1}) — 下一状态的价值估计
                        terminated=True → 0.0
                        truncated=True → V(final_obs)
                        正常步 → V(next_state)
        """
        self.rewards.append(reward)
        self.terminated_list.append(terminated)
        self.dones.append(terminated or truncated)
        self.next_values.append(next_value)

    def act(self, obs: np.ndarray, train: bool = True) -> int:
        """
        根据当前观测选择动作。

        训练模式下，记录状态和动作用于后续 n-step 更新。
        log_prob 和 value 在 update() 中重算以保留计算图。

        Args:
            obs: 观测/状态向量
            train: 训练模式或评估模式

        Returns:
            动作索引
        """
        obs_t = (
            torch.FloatTensor(np.array(obs, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

        if train:
            with torch.no_grad():
                logits, _ = self.actor_critic(obs_t)
                log_prob_all = torch.log_softmax(logits, dim=-1)
                probs = torch.exp(log_prob_all)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

            self.states.append(obs_t.squeeze(0))
            self.actions.append(action)
            return action
        else:
            with torch.no_grad():
                logits, _ = self.actor_critic(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            return action

    def update(self) -> Dict[str, float]:
        """
        使用 rollout 缓存中的 n_step 数据更新网络。

        计算流程：
        1. 在完整的状态序列上做一次前向传播（保留计算图）
        2. 调用 compute_n_step_returns() 计算 n-step 回报
        3. 优势 A = R_t - V(s_t)
        4. 策略损失 + 价值损失 + 熵奖励 → 总损失

        Returns:
            {"policy_loss": float, "value_loss": float, "entropy": float}
        """
        T = len(self.states)
        if T == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # 一致性检查
        for name, lst in [("actions", self.actions), ("rewards", self.rewards),
                          ("dones", self.dones), ("terminated_list", self.terminated_list),
                          ("next_values", self.next_values)]:
            if len(lst) != T:
                raise RuntimeError(
                    f"Rollout buffer mismatch: states={T} but {name}={len(lst)}"
                )

        # 转换数据为张量
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)

        # Actor-Critic 前向传播（保留计算图用于梯度）
        logits, values = self.actor_critic(states)
        values = values.squeeze(-1)

        # 计算 log_prob 和熵
        log_prob_all = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_prob_all)
        action_log_probs = log_prob_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_prob_all).sum(dim=-1).mean()

        # 使用公共函数计算 n-step 回报
        returns_np = compute_n_step_returns(
            self.rewards, self.dones, self.terminated_list,
            self.next_values, self.gamma,
        )
        returns = torch.FloatTensor(returns_np).to(self.device)

        # 计算优势
        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (
                (advantages - advantages.mean())
                / (advantages.std(unbiased=False) + 1e-8)
            )

        # 损失计算
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

        # 清空缓存
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.terminated_list.clear()
        self.next_values.clear()

        return metrics
