"""
深度 Q 网络（DQN）及其变体

包含:
- DQNConfig: 超参数配置数据类
- DQNAgent: 标准 DQN，带经验回放和目标网络
- DoubleDQNAgent: Double DQN，online 网络选动作，target 网络估值
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_course.agents.base import BaseAgent
from rl_course.buffers.replay_buffer import ReplayBuffer
from rl_course.networks.mlp import QNetwork


@dataclass
class DQNConfig:
    """DQN 超参数配置数据类

    可通过数据类统一管理超参数，方便序列化和实验配置。

    Attributes:
        state_dim: 状态空间维度
        n_actions: 动作空间大小（离散动作数）
        hidden_dims: 隐藏层维度列表
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最小探索率
        epsilon_decay: 探索率衰减系数（每次 act 后衰减）
        lr: 学习率
        batch_size: 训练批次大小
        buffer_capacity: 经验回放缓冲区容量
        target_update_freq: 目标网络更新频率（步数）
        device: 计算设备
        use_huber: 是否使用 Huber 损失（SmoothL1Loss），否则使用 MSE
        seed: 随机种子
    """
    state_dim: int
    n_actions: int
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 10000
    target_update_freq: int = 100
    device: str = "cpu"
    use_huber: bool = True
    seed: Optional[int] = None


class DQNAgent(BaseAgent):
    """标准 DQN 智能体

    使用经验回放（Experience Replay）和目标网络（Target Network）稳定训练。
    核心思想：通过采样历史经验打破数据相关性，使用固定目标网络减少训练震荡。

    Args:
        state_dim: 状态空间维度
        n_actions: 动作空间大小
        hidden_dims: Q 网络隐藏层维度，默认 [64, 64]
        gamma: 折扣因子
        epsilon_start: 初始 epsilon 值
        epsilon_end: 最小 epsilon 值
        epsilon_decay: epsilon 衰减率
        lr: 学习率
        batch_size: 训练批次大小
        buffer_capacity: 回放缓冲区容量
        target_update_freq: 目标网络更新频率
        device: 设备
        use_huber: 是否使用 Huber 损失
        seed: 随机种子
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = None,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
        device: str = "cpu",
        use_huber: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)

        # ========== 超参数 ==========
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 64]
        self.gamma = gamma
        self.epsilon = epsilon_start          # 当前探索率
        self.epsilon_start = epsilon_start    # 初始探索率（用于恢复）
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.use_huber = use_huber

        # ========== Q 网络 ==========
        # online Q 网络：当前训练的 Q 函数，参数实时更新
        self.q_network = QNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # target Q 网络：用于计算 TD 目标，参数周期性从 online 网络复制
        # 目的是固定目标分布，减少训练震荡
        self.target_network = QNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # 初始化 target 网络参数与 online 网络一致
        self.target_network.load_state_dict(self.q_network.state_dict())

        # ========== 优化器 ==========
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # ========== 损失函数 ==========
        # Huber (SmoothL1) 损失对异常值更鲁棒，训练更稳定
        self.criterion: nn.Module
        if use_huber:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

        # ========== 经验回放缓冲区 ==========
        # 存储 (s, a, r, s', done) 元组，打破数据时序相关性
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            device=device,
        )

        # ========== 训练状态 ==========
        self.learn_step_counter = 0     # 学习步数计数（用于触发生成网络更新）
        self.training_steps = 0         # 总训练步数

        # ========== 历史指标 ==========
        self.loss_history: List[float] = []      # 损失历史记录
        self.q_value_history: List[float] = []   # Q 值历史记录

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """预处理状态为 float32 张量

        确保状态数据格式统一，便于网络前向传播。

        Args:
            state: 原始状态
                   - 单样本: shape (state_dim,)
                   - 批量:   shape (batch_size, state_dim)

        Returns:
            float32 张量
                - 单样本: shape (1, state_dim)   — 添加 batch 维度
                - 批量:   shape (batch_size, state_dim)
        """
        # 确保数据类型为 float32
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(np.asarray(state, dtype=np.float32))

        # 单样本添加 batch 维度: (state_dim,) -> (1, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        return state.to(self.device)

    # ------------------------------------------------------------------
    # 动作选择
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, train: bool = True) -> int:
        """epsilon-贪心策略选择动作

        训练模式: 以 epsilon 概率随机探索，否则选择 Q 值最大的动作。
        评估模式: 始终选择 Q 值最大的动作（确定性策略）。

        Args:
            obs: 观测/状态，shape (state_dim,)
            train: 是否使用 epsilon-贪心（训练模式）

        Returns:
            动作索引，int 类型，范围 [0, n_actions)
        """
        # 训练模式且满足探索条件时随机探索
        if train and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            # 利用：选择 Q 值最大的动作
            # state_tensor: (1, state_dim)
            state_tensor = self._preprocess_state(obs)

            with torch.no_grad():
                q_values = self.q_network(state_tensor)  # shape: (1, n_actions)
                action = int(q_values.argmax(dim=1).item())

        # 训练模式下衰减探索率
        if train:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return action

    # ------------------------------------------------------------------
    # TD 目标计算
    # ------------------------------------------------------------------

    def _compute_target(
        self,
        rewards: torch.Tensor,      # shape: (batch_size,)
        next_states: torch.Tensor,  # shape: (batch_size, state_dim)
        terminated: torch.Tensor,   # shape: (batch_size,) — 1=真正终止, 0=未终止或截断
    ) -> torch.Tensor:
        """计算 TD 目标（标准 DQN）

        DQN 使用 target 网络中最大的 Q 值作为未来回报估计:
            target = r + gamma * max_a' Q_target(s', a') * (1 - terminated)

        关键区分:
        - terminated = 1 (真正到达目标): 不 bootstrap, 未来价值为 0
        - terminated = 0 (未终止或时间截断): 从 next_state bootstrap
          即使 episode 因时间截断而结束, 状态仍有有效的未来价值

        Args:
            rewards: 奖励，shape (batch_size,)
            next_states: 下一状态，shape (batch_size, state_dim)
            terminated: 真正终止标志 (非 done), shape (batch_size,)

        Returns:
            TD 目标值，shape (batch_size,)
        """
        with torch.no_grad():
            # 使用目标网络计算下一状态的 Q 值
            next_q_values = self.target_network(next_states)  # (batch, n_actions)

            # 取所有动作中的最大 Q 值（标准 DQN 的过估计来源）
            max_next_q = next_q_values.max(dim=1)[0]          # (batch,)

            # TD 目标: 真正终止时不 bootstrap, 截断时仍然 bootstrap
            target = rewards + self.gamma * max_next_q * (1.0 - terminated)

        return target

    # ------------------------------------------------------------------
    # 参数更新
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """执行一次参数更新

        核心流程:
        1. 从经验回放缓冲区采样一个 minibatch
        2. 计算当前 Q(s, a) —— online 网络前向传播
        3. 计算 TD 目标 —— target 网络 + 贝尔曼方程
        4. 计算损失（MSE 或 Huber）
        5. 梯度下降更新 online 网络参数
        6. 定期硬复制参数到 target 网络

        Returns:
            {
                "loss": float,    — 当前批次的损失值
                "epsilon": float, — 当前探索率
                "avg_q": float,   — 当前批次的平均 Q 值
            }
        """
        # 缓冲区数据不足时跳过更新
        if len(self.replay_buffer) < self.batch_size:
            return {
                "loss": 0.0,
                "epsilon": self.epsilon,
                "avg_q": 0.0,
            }

        # ----- 1. 采样 -----
        # 从经验回放中均匀随机采样一个 minibatch
        # states:      (batch_size, state_dim)
        # actions:     (batch_size,)
        # rewards:     (batch_size,)
        # next_states: (batch_size, state_dim)
        # dones:       (batch_size,) — episode 边界 (terminated or truncated)
        # terminated:  (batch_size,) — 真正的环境终止
        states, actions, rewards, next_states, dones, terminated = \
            self.replay_buffer.sample(self.batch_size)

        # ----- 2. 计算当前 Q 值 -----
        # Q(s, a): 对每个样本，选中实际执行动作对应的 Q 值
        q_values = self.q_network(states)                       # (batch, n_actions)
        q_value = q_values.gather(1, actions.unsqueeze(1))      # (batch, 1)
        q_value = q_value.squeeze(1)                            # (batch,)

        # ----- 3. 计算 TD 目标 -----
        # 使用 terminated (而非 dones) 作为 bootstrap mask
        # 时间截断的 transition 仍然需要从 next_state bootstrap
        target = self._compute_target(rewards, next_states, terminated)  # (batch,)

        # ----- 4. 计算损失 -----
        # MSE 或 Huber 损失：衡量当前 Q 值与 TD 目标之间的差距
        loss = self.criterion(q_value, target)

        # ----- 5. 梯度下降 -----
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪：防止梯度爆炸，将梯度范数限制在 10.0 以内
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # ----- 6. 记录指标 -----
        self.learn_step_counter += 1
        self.training_steps += 1

        avg_q = q_value.mean().item()
        loss_val = loss.item()

        self.loss_history.append(loss_val)
        self.q_value_history.append(avg_q)

        # ----- 7. 定期更新目标网络 -----
        # 硬更新：直接将 online 网络的参数复制给 target 网络
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": loss_val,
            "epsilon": self.epsilon,
            "avg_q": avg_q,
        }

    # ------------------------------------------------------------------
    # 保存与加载
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存模型检查点到磁盘

        保存内容包括:
        - Q 网络和目标网络的参数
        - 优化器状态
        - 当前 epsilon 值和学习步数
        - 超参数配置（用于加载时重建）

        Args:
            path: 保存路径（通常为 .pth 或 .pt 文件）
        """
        checkpoint = {
            # 网络参数
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            # 优化器状态
            "optimizer_state_dict": self.optimizer.state_dict(),
            # 训练状态
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
            "training_steps": self.training_steps,
            # 超参数（用于检查兼容性）
            "config": {
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "buffer_capacity": self.buffer_capacity,
                "target_update_freq": self.target_update_freq,
                "use_huber": self.use_huber,
            },
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """从磁盘加载模型检查点

        恢复网络参数、优化器状态和训练状态。

        Args:
            path: 检查点文件路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 恢复训练状态（使用 get 兼容旧版本检查点）
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.learn_step_counter = checkpoint.get("learn_step_counter", 0)
        self.training_steps = checkpoint.get("training_steps", 0)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: DQNConfig) -> "DQNAgent":
        """从 DQNConfig 数据类构造智能体

        Args:
            config: DQN 超参数配置

        Returns:
            DQNAgent 实例
        """
        return cls(
            state_dim=config.state_dim,
            n_actions=config.n_actions,
            hidden_dims=config.hidden_dims,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            lr=config.lr,
            batch_size=config.batch_size,
            buffer_capacity=config.buffer_capacity,
            target_update_freq=config.target_update_freq,
            device=config.device,
            use_huber=config.use_huber,
            seed=config.seed,
        )


class DoubleDQNAgent(DQNAgent):
    """Double DQN 智能体

    解决标准 DQN 中的 Q 值过估计问题（overestimation bias）。
    核心思想是将"动作选择"和"动作评估"解耦:
    - 使用 online 网络选择最优动作: a' = argmax_a' Q_online(s', a')
    - 使用 target 网络评估该动作的值: Q_target(s', a')

    标准 DQN 使用相同的 max 算子同时做选择和评估，导致正偏差。
    Double DQN 通过解耦有效缓解了这一问题。

    与 DQNAgent 的唯一区别在于 _compute_target 方法的实现。
    """

    def _compute_target(
        self,
        rewards: torch.Tensor,      # shape: (batch_size,)
        next_states: torch.Tensor,  # shape: (batch_size, state_dim)
        terminated: torch.Tensor,   # shape: (batch_size,) — 1=真正终止, 0=未终止或截断
    ) -> torch.Tensor:
        """计算 TD 目标（Double DQN）

        公式:
            a* = argmax_a Q_online(s', a)
            target = r + gamma * Q_target(s', a*) * (1 - terminated)

        Args:
            rewards: 奖励，shape (batch_size,)
            next_states: 下一状态，shape (batch_size, state_dim)
            terminated: 真正终止标志 (非 dones), shape (batch_size,)

        Returns:
            TD 目标值，shape (batch_size,)
        """
        with torch.no_grad():
            online_q = self.q_network(next_states)   # (batch, n_actions)
            best_actions = online_q.argmax(dim=1)     # (batch,)

            target_q = self.target_network(next_states)
            max_next_q = target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # 使用 terminated (而非 dones): 截断时仍需 bootstrap
            target = rewards + self.gamma * max_next_q * (1.0 - terminated)

        return target

    @classmethod
    def from_config(cls, config: DQNConfig) -> "DoubleDQNAgent":
        """从 DQNConfig 数据类构造智能体

        Args:
            config: DQN 超参数配置

        Returns:
            DoubleDQNAgent 实例
        """
        return cls(
            state_dim=config.state_dim,
            n_actions=config.n_actions,
            hidden_dims=config.hidden_dims,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            lr=config.lr,
            batch_size=config.batch_size,
            buffer_capacity=config.buffer_capacity,
            target_update_freq=config.target_update_freq,
            device=config.device,
            use_huber=config.use_huber,
            seed=config.seed,
        )
