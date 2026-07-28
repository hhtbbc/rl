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
        action = agent.act(obs)                          # 选择动作
        agent.store(reward, terminated, truncated,
                    next_value)                           # 存储环境反馈
        metrics = agent.update()                          # n-step 更新

    Episode boundary 处理:
        - 每一步存储 next_value = V(s_{t+1}) (在 env.reset() 之前计算)
        - done=True 处: G = r + γ·(1-terminated)·next_value (不跨 episode)
        - done=False 处: G = r + γ·G (正常累积)
        - 两个 mask: terminated 控制 bootstrap, done 控制跨 episode 传播

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
    ):
        super().__init__(seed=None)
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
        next_value: float = 0.0,
    ) -> None:
        """
        存储环境反馈。

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

    def update(self) -> Dict[str, float]:
        """
        使用 rollout 缓存中的 n_step 数据更新网络。

        计算流程：
        1. 在完整的状态序列上做一次 Actor-Critic 前向传播（保留计算图）
        2. 计算 n-step 回报，处理 episode boundary
        3. 优势 A = R_t - V(s_t)
        4. 策略损失: -mean(log_prob * A.detach())
        5. 价值损失: MSE(V(s), R)
        6. 熵奖励: entropy_coef * mean(H(pi))

        Episode boundary 处理（正确实现）:
        - 反向遍历时，先检查 done[t] 再累积 G
        - done=True 且 terminated=False (truncated):
          G = r + γ·next_value (bootstrap from V, 不跨 episode)
        - done=True 且 terminated=True:
          G = r (不 bootstrap, future value=0)
        - done=False:
          G = r + γ·G (正常累积同一 episode 内的回报)

        Returns:
            {"policy_loss": float, "value_loss": float, "entropy": float}
        """
        if len(self.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        T = len(self.states)

        # 转换数据为张量
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)

        # Actor-Critic 前向传播（保留计算图用于梯度）
        logits, values = self.actor_critic(states)
        values = values.squeeze(-1)  # shape: (T,)

        # 计算 log_prob 和熵（数值稳定）
        log_prob_all = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_prob_all)

        # 选中动作的对数概率
        action_log_probs = log_prob_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # 策略熵: H(pi) = -sum pi(a|s) * log pi(a|s)
        entropy = -(probs * log_prob_all).sum(dim=-1).mean()

        # ================================================================
        # 计算 n-step 回报 (正确处理 episode boundary)
        #
        # 关键: done[t] 必须在 G 被使用之前检查，否则会跨 episode 污染。
        #
        # 例如 rollout: [r0, r1, r2(done), r3, r4]
        #   反向遍历到 r2 时，当前 G 不应包含 r3, r4 的贡献。
        #   正确做法: 先检查 done[2]=True, 重置 G, 再计算 R2。
        # ================================================================
        returns = np.zeros(T, dtype=np.float32)

        # 初始化 G: 如果 rollout 末尾不是 episode 边界，
        # 从最后一个 next_value (V(s_T)) 开始 bootstrap
        if self.dones[-1]:
            G = 0.0
        else:
            G = self.next_values[-1]

        for t in reversed(range(T)):
            if self.dones[t]:
                # Episode 边界: 回报仅包含本 transition
                # terminated=False (truncated): bootstrap from V(s_{t+1})
                # terminated=True: future value = 0
                bootstrap_mask = 1.0 - float(self.terminated_list[t])
                G = self.rewards[t] + self.gamma * self.next_values[t] * bootstrap_mask
            else:
                G = self.rewards[t] + self.gamma * G
            returns[t] = G

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
        self.next_values.clear()

        return {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "entropy": entropy_val,
        }
