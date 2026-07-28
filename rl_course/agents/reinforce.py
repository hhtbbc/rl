"""
REINFORCE 算法实现族

包含：
- REINFORCEAgent: 基础 REINFORCE (Monte Carlo Policy Gradient)
- REINFORCEWithBaselineAgent: 带基线的 REINFORCE，使用学习的价值函数作为基线
"""

from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_course.agents.base import BaseAgent
from rl_course.networks.mlp import PolicyNetwork, ValueNetwork


class REINFORCEAgent(BaseAgent):
    """
    基础 REINFORCE 智能体 (Monte Carlo Policy Gradient)。

    使用完整 episode 的蒙特卡洛回报来更新策略网络。
    策略梯度定理: nabla J(theta) = E[nabla log pi_theta(a|s) * G_t]
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        gamma: float = 0.99,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: 状态空间维度
            n_actions: 离散动作数量
            hidden_dims: 隐藏层维度列表
            gamma: 折扣因子
            lr: 学习率
            device: 设备 ("cpu" 或 "cuda")
        """
        super().__init__(seed=None)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)

        # 策略网络 pi(a|s): 输入状态，输出动作概率分布
        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode 经验缓存
        # 每次 act() 记录一步数据，update() 时统一处理整个 episode
        self.episode_states: List[torch.Tensor] = []   # 状态序列
        self.episode_actions: List[int] = []            # 动作序列
        self.episode_rewards: List[float] = []          # 奖励序列（由外部在 env.step 后追加）
        self.episode_log_probs: List[torch.Tensor] = [] # 对数概率序列

    def act(self, obs: np.ndarray, train: bool = True) -> int:
        """
        根据当前观测选择动作。

        训练模式下从策略分布采样并记录 log_prob（保留计算图供后续梯度计算）；
        评估模式下选取概率最大的动作。

        Args:
            obs: 观测/状态向量
            train: 训练模式（随机采样）或评估模式（贪心）

        Returns:
            动作索引
        """
        # 将 numpy 数组转为 tensor，添加 batch 维度
        obs_t = (
            torch.FloatTensor(np.array(obs, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

        if train:
            # 训练模式：采样动作并记录 log_prob
            # 从策略网络获取 logits（未经 softmax 的原始分数）
            logits = self.policy.get_logits(obs_t)
            # 使用 log_softmax 保证数值稳定性（避免概率接近 0 时的 log 爆炸）
            log_prob_all = torch.log_softmax(logits, dim=-1)
            # 转换为概率用于采样
            probs = torch.exp(log_prob_all)

            # 从分类分布中采样一个动作
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # 获取选中动作对应的 log_prob
            action_log_prob = log_prob_all.gather(1, action.unsqueeze(-1)).squeeze(-1)

            # 记录到 episode 缓存
            self.episode_states.append(obs_t.squeeze(0))
            self.episode_actions.append(action.item())
            self.episode_log_probs.append(action_log_prob)

            return action.item()
        else:
            # 评估模式：贪心选取最优动作
            with torch.no_grad():
                probs = self.policy(obs_t)
                action = torch.argmax(probs, dim=-1).item()
            return action

    def update(self) -> Dict[str, float]:
        """
        使用完整 episode 数据计算策略梯度并更新网络。

        计算步骤：
        1. 计算折扣回报 G_t = sum gamma^{k-t} r_k
        2. 归一化回报以减小方差
        3. 计算策略损失 L = -mean(log_prob * G)
        4. 反向传播更新策略网络

        Returns:
            {"policy_loss": float, "episode_return": float}
        """
        if len(self.episode_states) == 0:
            return {"policy_loss": 0.0, "episode_return": 0.0}

        # 组装 episode 数据
        # log_probs 保留计算图，梯度可回传到策略网络
        states = torch.stack(self.episode_states).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        rewards = torch.FloatTensor(self.episode_rewards).to(self.device)
        log_probs = torch.cat(self.episode_log_probs).to(self.device)

        # 计算折扣回报 G_t
        # 从后向前递推: G_t = r_t + gamma * G_{t+1}
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # 归一化回报（减小方差，稳定训练）
        # 注意: 这是训练启发式 (heuristic)，并非原始 REINFORCE 的必要部分
        # 原始 REINFORCE 使用未归一化的 G_t，是无偏估计
        # 归一化引入了 episode 间的相关性，但实践中显著降低方差
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 策略梯度损失
        # L = -1/N * sum log pi(a_t|s_t) * G_t
        policy_loss = -(log_probs * returns).mean()

        # 更新策略网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # 记录指标
        episode_return = rewards.sum().item()
        policy_loss_val = policy_loss.item()

        # 清空 episode 缓存，准备下一 episode
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()

        return {
            "policy_loss": policy_loss_val,
            "episode_return": episode_return,
        }


class REINFORCEWithBaselineAgent(BaseAgent):
    """
    带基线的 REINFORCE 智能体。

    使用学习的价值函数 V(s) 作为基线来降低策略梯度的方差。
    优势函数: A_t = G_t - V(s_t)
    策略梯度: nabla J(theta) = E[nabla log pi_theta(a|s) * A_t]
    价值损失: L_V = MSE(V(s_t), G_t)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        gamma: float = 0.99,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: 状态空间维度
            n_actions: 离散动作数量
            hidden_dims: 隐藏层维度列表
            gamma: 折扣因子
            lr: 学习率
            device: 设备 ("cpu" 或 "cuda")
        """
        super().__init__(seed=None)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)

        # 策略网络 pi(a|s)
        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        # 价值网络 V(s) — 作为基线，降低策略梯度方差
        self.value = ValueNetwork(state_dim, hidden_dims).to(self.device)
        # 联合优化器（同时优化策略和价值网络）
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

        # Episode 经验缓存
        self.episode_states: List[torch.Tensor] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_log_probs: List[torch.Tensor] = []

    def act(self, obs: np.ndarray, train: bool = True) -> int:
        """
        根据当前观测选择动作。

        Args:
            obs: 观测/状态向量
            train: 训练模式（采样）或评估模式（贪心）

        Returns:
            动作索引
        """
        obs_t = (
            torch.FloatTensor(np.array(obs, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )

        if train:
            # 获取 logits 并计算 log_softmax（数值稳定）
            logits = self.policy.get_logits(obs_t)
            log_prob_all = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_prob_all)

            # 采样动作
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
            action_log_prob = log_prob_all.gather(1, action.unsqueeze(-1)).squeeze(-1)

            # 存储
            self.episode_states.append(obs_t.squeeze(0))
            self.episode_actions.append(action.item())
            self.episode_log_probs.append(action_log_prob)

            return action.item()
        else:
            with torch.no_grad():
                probs = self.policy(obs_t)
                action = torch.argmax(probs, dim=-1).item()
            return action

    def update(self) -> Dict[str, float]:
        """
        使用带基线的策略梯度更新网络。

        计算步骤：
        1. 计算折扣回报 G_t
        2. 通过价值网络计算基线 V(s_t)
        3. 计算优势 A_t = G_t - V(s_t)
        4. 策略损失: L_policy = -mean(log_prob * A.detach())  <- detach 至关重要！
        5. 价值损失: L_value = MSE(V(s_t), G_t)
        6. 反向传播更新所有网络

        Returns:
            {"policy_loss": float, "value_loss": float, "episode_return": float}
        """
        if len(self.episode_states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "episode_return": 0.0}

        # 组装 episode 数据
        states = torch.stack(self.episode_states).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        rewards = torch.FloatTensor(self.episode_rewards).to(self.device)
        log_probs = torch.cat(self.episode_log_probs).to(self.device)

        # 计算折扣回报 G_t (原始尺度，不归一化)
        # Critic 必须学习预测原始尺度的 V^π(s)，而非归一化后的值
        raw_returns_list = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            raw_returns_list.insert(0, G)
        raw_returns = torch.FloatTensor(raw_returns_list).to(self.device)

        # 计算价值基线 V(s_t) — Critic 学习拟合原始回报
        values = self.value(states).squeeze(-1)  # shape (T,)

        # 计算优势 A_t = G_t - V(s_t)（在原始回报尺度上）
        advantages = raw_returns - values  # shape (T,)

        # 标准化优势 (Advantage Normalization) — 降低策略梯度方差
        # 注意: 标准化的是优势，不是回报！
        # Critic 的 target 仍然是原始回报 (见 value_loss)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ===============================================================
        # 关键：对 advantages 调用 detach()！
        #
        # 优势 A_t = G_t - V(s_t) 的计算图中包含 V(s_t)，
        # 所以 A_t 张量带有价值网络的 grad_fn。
        # 如果策略损失 -mean(log_prob * A) 反向传播时梯度经过 A，
        # 会错误地更新价值网络的参数——价值网络应该只通过
        # value_loss (MSE) 来学习拟合回报。
        #
        # detach() 切断优势张量中与价值网络相连的计算路径，
        # 使策略梯度只影响策略网络参数。
        # 不 detach 会导致价值网络发散，训练失败。
        # ===============================================================
        policy_loss = -(log_probs * advantages.detach()).mean()

        # 价值损失：Critic 拟合原始（未归一化）回报
        value_loss = F.mse_loss(values, raw_returns)

        # 总损失
        total_loss = policy_loss + value_loss

        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 记录指标
        episode_return = rewards.sum().item()
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()

        # 清空 episode 缓存
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()

        return {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "episode_return": episode_return,
        }
