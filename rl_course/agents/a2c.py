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


class A2CAgent(BaseAgent):
    """
    Single-environment n-step Advantage Actor-Critic。

    注意: 这是单环境版本，用于教学。标准 A2C 使用多个同步环境并行采样。
    本实现演示 n-step return、advantage、entropy bonus 等核心概念。

    API:
        action = agent.act(obs)     # 选择动作
        agent.store(r, term, trunc) # 存储环境反馈
        metrics = agent.update(next_obs) # n-step 更新

    Loss = L_policy + L_value - entropy_coef * H(pi)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        gamma: float = 0.99,
        lr: float = 1e-3,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__(seed=None)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        self.actor_critic = ActorCriticNetwork(
            state_dim, n_actions, hidden_dims
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Rollout 缓存
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []             # episode 边界
        self.terminated_list: List[bool] = []   # 真正终止

    def store(
        self, reward: float, terminated: bool, truncated: bool
    ) -> None:
        """
        存储环境反馈。

        Args:
            reward: 即时奖励
            terminated: 环境真正终止
            truncated: 时间截断
        """
        self.rewards.append(reward)
        self.terminated_list.append(terminated)
        self.dones.append(terminated or truncated)

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
            # 训练模式：使用 no_grad 提高效率
            # log_prob 和 value 在 update 中重新前向计算以保留 grad
            with torch.no_grad():
                logits, _ = self.actor_critic(obs_t)
                # 数值稳定的概率计算
                log_prob_all = torch.log_softmax(logits, dim=-1)
                probs = torch.exp(log_prob_all)

                # 从策略分布采样动作
                action = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

            # 记录状态和动作（log_prob 在 update 时重新计算）
            self.states.append(obs_t.squeeze(0))
            self.actions.append(action)

            return action
        else:
            # 评估模式：贪心选择
            with torch.no_grad():
                logits, _ = self.actor_critic(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            return action

    def update(self, next_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        使用 rollout 缓存中的 n_step 数据更新网络。

        计算流程：
        1. 在完整的状态序列上做一次 Actor-Critic 前向传播（保留计算图）
        2. 计算 n-step 回报 R_t^{(n)}，使用 V(s_{t+n}) 引导
        3. 优势 A = R_t - V(s_t)
        4. 策略损失: -mean(log_prob * A.detach())
        5. 价值损失: MSE(V(s), R)
        6. 熵奖励: entropy_coef * mean(H(pi))

        Args:
            next_obs: 最后一个动作后的下一个观测，用于 n-step 引导。
                      None 表示 episode 已结束（引导值为 0）。

        Returns:
            {"policy_loss": float, "value_loss": float, "entropy": float}
        """
        if len(self.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # 转换数据为张量
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)

        # Actor-Critic 前向传播（保留计算图用于梯度）
        logits, values = self.actor_critic(states)
        values = values.squeeze(-1)  # shape: (n_steps,)

        # 计算 log_prob 和熵（数值稳定）
        log_prob_all = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_prob_all)

        # 选中动作的对数概率
        action_log_probs = log_prob_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # 策略熵: H(pi) = -sum pi(a|s) * log pi(a|s)
        entropy = -(probs * log_prob_all).sum(dim=-1).mean()

        # 计算 n-step 引导值 V(s_{t+n})
        # 关键区分:
        #   - 真正终止 (terminated): 不 bootstrap, V=0
        #   - 时间截断 (truncated, done=True but terminated=False): 需要 bootstrap
        last_terminated = self.terminated_list[-1] if self.terminated_list else False
        if next_obs is not None and not last_terminated:
            # episode 未真正终止 (可能是截断), 使用 V(s_{t+n}) 引导
            next_obs_t = (
                torch.FloatTensor(np.array(next_obs, dtype=np.float32))
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                _, bootstrap_value = self.actor_critic(next_obs_t)
            G = bootstrap_value.squeeze(-1).item()
        else:
            # episode 真正终止, 引导值为 0
            G = 0.0

        # 从后向前递归计算 n-step 回报
        # 使用 terminated (而非 done) 控制 bootstrap:
        #   - terminated=True: G = r (不 bootstrap)
        #   - terminated=False (包括 truncated): G = r + gamma * G (bootstrap)
        # 注意: 这是教学简化。若 rollout 中间发生 truncation 且 env 已 reset，
        # G 会跨 episode 边界传播。标准做法是保存 final_observation。
        returns = []
        for i in reversed(range(len(rewards))):
            term = self.terminated_list[i]
            G = rewards[i] + self.gamma * G * (1.0 - float(term))
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # 计算优势 A = R_t - V(s_t)
        advantages = returns - values
        # 归一化优势（减小方差，稳定训练）
        if advantages.numel() > 1:
            advantages = (
                (advantages - advantages.mean())
                / (advantages.std(unbiased=False) + 1e-8)
            )

        # ===============================================================
        # detach 优势：阻止策略梯度反向传播影响 Critic 和价值头。
        # ===============================================================
        # 策略损失（优势 detached，梯度只更新 Actor）
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        # 价值损失（使用原始 returns，不归一化）
        value_loss = F.mse_loss(values, returns)

        # 总损失 = 策略损失 + 价值损失 - 熵奖励
        # 减熵奖励是因为 maximize entropy（鼓励探索）
        total_loss = (
            policy_loss + value_loss - self.entropy_coef * entropy
        )

        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 记录指标
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        entropy_val = entropy.item()

        # 清空 rollout 缓存
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.terminated_list.clear()

        return {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "entropy": entropy_val,
        }
