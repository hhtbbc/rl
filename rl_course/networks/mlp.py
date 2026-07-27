"""
多层感知机（MLP）网络模块

提供强化学习中常用的网络架构：
- MLP: 通用多层感知机
- ValueNetwork: 状态价值网络 V(s)
- QNetwork: 动作价值网络 Q(s, a)
- PolicyNetwork: 离散动作策略网络 π(a|s)
- ActorCriticNetwork: Actor-Critic 共享特征提取器
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> None:
    """
    正交初始化，有助于稳定训练。

    Args:
        layer: 线性层
        gain: 缩放因子（tanh 推荐 √2，ReLU 推荐 √2）
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """通用多层感知机

    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表，如 [64, 64]
        output_dim: 输出维度
        activation: 激活函数（"relu" 或 "tanh"）
        use_orthogonal_init: 是否使用正交初始化
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        if use_orthogonal_init:
            self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            orthogonal_init(module, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，shape (batch_size, input_dim)

        Returns:
            输出张量，shape (batch_size, output_dim)
        """
        return self.network(x)


class ValueNetwork(nn.Module):
    """状态价值网络 V(s)

    输出一个标量值，表示状态的期望回报。

    Args:
        state_dim: 状态维度
        hidden_dims: 隐藏层维度列表
        activation: 激活函数
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: shape (batch_size, state_dim)

        Returns:
            价值，shape (batch_size, 1)
        """
        return self.mlp(state)


class QNetwork(nn.Module):
    """动作价值网络 Q(s, a)

    输出每个动作的 Q 值。输入状态，输出所有动作的 Q 值向量。

    Args:
        state_dim: 状态维度
        n_actions: 动作数量
        hidden_dims: 隐藏层维度列表
        activation: 激活函数
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=n_actions,
            activation=activation,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: shape (batch_size, state_dim)

        Returns:
            Q 值，shape (batch_size, n_actions)
        """
        return self.mlp(state)


class PolicyNetwork(nn.Module):
    """离散动作策略网络 π(a|s)

    输出每个动作的概率（通过 softmax）。

    Args:
        state_dim: 状态维度
        n_actions: 动作数量
        hidden_dims: 隐藏层维度列表
        activation: 激活函数
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=n_actions,
            activation=activation,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: shape (batch_size, state_dim)

        Returns:
            动作概率分布，shape (batch_size, n_actions)
        """
        logits = self.mlp(state)
        return torch.softmax(logits, dim=-1)

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        """获取 softmax 之前的 logits（用于数值稳定计算 log_prob）"""
        return self.mlp(state)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic 共享特征提取器

    共享前几层，分别输出策略和价值。

    Args:
        state_dim: 状态维度
        n_actions: 动作数量
        hidden_dims: 共享隐藏层维度列表
        activation: 激活函数
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        # 共享特征提取器
        shared_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(act_fn)
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers)

        # Actor 头（策略）
        self.actor_head = nn.Linear(prev_dim, n_actions)

        # Critic 头（价值）
        self.critic_head = nn.Linear(prev_dim, 1)

        # 初始化
        self.apply(lambda m: orthogonal_init(m) if isinstance(m, nn.Linear) else None)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: shape (batch_size, state_dim)

        Returns:
            (action_logits, state_value)
            - action_logits: shape (batch_size, n_actions) — 未经过 softmax
            - state_value: shape (batch_size, 1)
        """
        features = self.shared(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_policy(self, state: torch.Tensor) -> torch.Tensor:
        """获取动作概率分布"""
        logits, _ = self.forward(state)
        return torch.softmax(logits, dim=-1)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        _, value = self.forward(state)
        return value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从策略中采样动作。

        Args:
            state: shape (batch_size, state_dim)
            deterministic: True 时选择概率最大的动作

        Returns:
            (action, log_prob, value)
            - action: shape (batch_size,) — 动作索引
            - log_prob: shape (batch_size,) — 所选动作的对数概率
            - value: shape (batch_size,) — 状态价值
        """
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # 计算所选动作的 log probability（数值稳定版）
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, action_log_prob, value.squeeze(-1)
