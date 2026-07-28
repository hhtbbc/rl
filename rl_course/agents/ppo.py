"""
PPO (Proximal Policy Optimization) 智能体
=============================================

PPO 是一种基于策略梯度的 on-policy 强化学习算法，通过 Clipped Surrogate Objective
限制每次更新的步长，在保证稳定性的同时兼顾样本效率。

核心创新:
    - Clipped Surrogate Objective: min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)
    - 多轮 Minibatch 更新: 每个 rollout 使用多轮小批量 SGD
    - GAE (Generalized Advantage Estimation): 平衡偏差与方差

Reference:
    Schulman et al., 2017 - Proximal Policy Optimization Algorithms
    https://arxiv.org/abs/1707.06347
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_course.agents.base import BaseAgent
from rl_course.networks.mlp import ActorCriticNetwork
from rl_course.buffers.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) 智能体。

    适用场景:
        - 离散动作空间
        - 需要稳定、高效策略梯度的任务

    算法流程:
        1. 使用当前策略收集 n_steps 步的轨迹数据
        2. 计算 GAE 优势估计和折扣回报
        3. 多轮小批量更新策略和价值网络
        4. 使用 Clipped Surrogate Objective 限制策略更新幅度

    Usage:
        agent = PPOAgent(state_dim=4, n_actions=2)
        obs, _ = env.reset()
        for step in range(agent.n_steps):
            action = agent.act(obs)             # 存储 state/action/log_prob/value
            next_obs, reward, done, _ = env.step(action)
            agent.store(reward, done)            # 存储 reward/done, 完成 transition
            obs = next_obs
            if done:
                obs, _ = env.reset()
        metrics = agent.update(last_obs=obs)     # 使用收集的数据更新网络
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 0.5,
        c2: float = 0.01,
        lr: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: 状态空间维度
            n_actions: 离散动作数量
            hidden_dims: 策略网络的隐藏层维度列表，默认 [64, 64]
            gamma: 折扣因子，默认 0.99
            gae_lambda: GAE λ 参数，用于平衡 TD 估计的偏差和方差，默认 0.95
            clip_epsilon: PPO 裁剪范围 ε，默认 0.2
            c1: 价值损失系数，默认 0.5
            c2: 熵正则系数，默认 0.01（鼓励探索）
            lr: Adam 优化器学习率，默认 3e-4
            n_steps: 每次更新前收集的步数（rollout 长度），默认 2048
            batch_size: 小批量大小，默认 64
            n_epochs: 每个 rollout 后的训练轮数，默认 10
            max_grad_norm: 梯度裁剪的最大 L2 范数，默认 0.5
            target_kl: 提前停止的 KL 散度阈值。当 minibatch 的近似 KL > target_kl 时
                       停止当前 epoch 的更新。None 表示不提前停止。
            device: 计算设备，默认 "cpu"
        """
        super().__init__()

        # ==================== 超参数 ====================
        self.state_dim: int = state_dim
        self.n_actions: int = n_actions
        self.hidden_dims: List[int] = hidden_dims
        self.gamma: float = gamma
        self.gae_lambda: float = gae_lambda
        self.clip_epsilon: float = clip_epsilon
        self.c1: float = c1
        self.c2: float = c2
        self.lr: float = lr
        self.n_steps: int = n_steps
        self.batch_size: int = batch_size
        self.n_epochs: int = n_epochs
        self.max_grad_norm: float = max_grad_norm
        self.target_kl: Optional[float] = target_kl
        self.device: torch.device = torch.device(device)

        # ==================== 策略网络 ====================
        # ActorCriticNetwork: 共享特征提取器 + 双头（Actor + Critic）
        #   forward(state) -> (logits, value)
        #     - logits: shape (batch_size, n_actions)，未经 softmax
        #     - value: shape (batch_size, 1)
        #   get_action(state, deterministic) -> (action, log_prob, value)
        #     - action: shape (batch_size,) — 动作索引
        #     - log_prob: shape (batch_size,) — 所选动作的 log 概率
        #     - value: shape (batch_size,) — 状态价值
        self.network: ActorCriticNetwork = ActorCriticNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # ==================== 优化器 ====================
        # Adam 是 PPO 常用的优化器，对超参数相对鲁棒
        self.optimizer: optim.Optimizer = optim.Adam(
            self.network.parameters(), lr=lr
        )

        # ==================== Rollout 缓冲区 ====================
        # 存储 n_steps 步的 (s, a, r, done, log_prob, value)
        # 用于 GAE 和多轮 minibatch 采样
        self.buffer: RolloutBuffer = RolloutBuffer(
            buffer_size=n_steps,
            state_dim=state_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
        )

        # ==================== 临时存储 ====================
        # act() 和 store() 之间的中间存储
        # act() 生成 state, action, log_prob, value
        # store() 补充 reward, done 后写入缓冲区
        self._last_state: Optional[np.ndarray] = None  # shape (state_dim,)
        self._last_action: Optional[int] = None
        self._last_log_prob: Optional[float] = None
        self._last_value: Optional[float] = None

        # ==================== 指标历史 ====================
        self._metrics_history: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
        }

    # ======================================================================
    # act() — 动作选择与数据暂存
    # ======================================================================
    def act(self, obs: np.ndarray, train: bool = True) -> int:
        """
        根据观测选择动作。

        训练模式 (train=True):
            - 使用当前策略随机采样动作（探索）
            - 将状态、动作、log_prob、value 暂存到临时变量
            - 等待后续 store() 补充 reward/done 后写入缓冲区

        评估模式 (train=False):
            - 选择概率最大的动作（确定性）
            - 不修改缓冲区

        Args:
            obs: 观测/状态向量，shape (state_dim,)
            train: 是否使用训练模式

        Returns:
            动作索引，范围 [0, n_actions)
        """
        # ---- 转换为 PyTorch 张量 ----
        # (1, state_dim): 添加 batch 维度以匹配网络输入格式
        obs_tensor: torch.Tensor = (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
        )  # shape: (1, state_dim)

        # ---- 前向传播 ----
        # get_action 内部调用 forward -> softmax -> multinomial 采样
        with torch.no_grad():
            action_tensor: torch.Tensor  # shape (1,)
            log_prob_tensor: torch.Tensor  # shape (1,)
            value_tensor: torch.Tensor  # shape (1,)
            action_tensor, log_prob_tensor, value_tensor = self.network.get_action(
                obs_tensor, deterministic=not train
            )

        # ---- 转为 Python 标量 ----
        action: int = int(action_tensor.item())

        if train:
            # ---- 训练模式: 暂存数据 ----
            # reward 和 done 在 env.step() 之后才可知，
            # 因此分两阶段存储: act() + store()
            self._last_state = obs
            self._last_action = action
            self._last_log_prob = float(log_prob_tensor.item())
            self._last_value = float(value_tensor.item())

        return action

    # ======================================================================
    # store() — 完成 transition 存储
    # ======================================================================
    def store(self, reward: float, done: bool, terminated: bool = False) -> None:
        """
        存储一步环境反馈，完成 transition 记录。

        与 act() 配对使用:
            1. act(obs)              — 获取动作，暂存 state/action/log_prob/value
            2. env.step(action)       — 执行动作，获得 reward, terminated, truncated
            3. store(r, done, term)   — 补充 reward/done/terminated，写入缓冲区

        Args:
            reward: 环境返回的即时奖励
            done: 当前 episode 是否结束 (terminated or truncated)
            terminated: 是否为真正的环境终止 (vs 时间截断)
                         True → GAE 中不 bootstrap（未来价值为 0）
                         False + done=True → 时间截断，GAE 中仍需 bootstrap
        """
        if self._last_state is None:
            raise RuntimeError(
                "store() 被调用前必须先调用 act()。"
                "调用顺序应为: act(obs) -> env.step(action) -> store(reward, done)"
            )

        # 将完整的 transition 写入缓冲区
        self.buffer.add(
            self._last_state,
            self._last_action,
            reward,
            done,
            self._last_log_prob,
            self._last_value,
            terminated=terminated,
        )

        # 清空临时存储，防止重复使用
        self._last_state = None

    # ======================================================================
    # update() — PPO 核心更新逻辑
    # ======================================================================
    def update(self, last_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        执行一次完整的 PPO 更新。

        更新步骤:
            1. 获取最后状态的价值（用于 GAE 引导）
            2. 计算 GAE 优势估计和折扣回报
            3. 标准化优势值
            4. 多轮 minibatch 优化:
               - 计算 Clipped Surrogate Objective
               - 计算 Clipped Value Loss
               - 计算熵正则项
               - 反向传播 + 梯度裁剪
            5. 跟踪指标（KL 散度、裁剪比例、解释方差等）
            6. 清空缓冲区

        Args:
            last_obs: 收集完 n_steps 步后环境的当前观测（即 s_{n+1}）。
                      用于 GAE 最后一步的 bootstrapping 价值 V(s_{n+1})。
                      如果为 None，假设最后一步为终止状态（V=0）。

        Returns:
            包含平均损失和指标的字典:
            - policy_loss:  策略裁剪损失
            - value_loss:   价值函数损失（裁剪后）
            - entropy_loss: 熵损失（取负后，-c2*entropy 鼓励探索）
            - approx_kl:    平均近似 KL 散度
            - clip_fraction: 被裁剪的比率占比（0~1）
            - explained_variance: 解释方差，衡量价值函数拟合质量
            - epochs_completed: 实际完成的 epoch 数
            - early_stopped: 是否因 KL 阈值提前停止
        """
        # ----------------------------------------------------------------
        # Step 1: 计算 GAE 需要的最后一步价值
        # ----------------------------------------------------------------
        # last_obs 是环境执行完 n_steps 步后的状态 s_{n+1}
        # 它的价值 V(s_{n+1}) 用于递推计算 GAE 最后一步的 TD 残差:
        #   δ_{n-1} = r_{n-1} + γ * V(s_n) * (1 - done_{n-1}) - V(s_{n-1})
        # 当 s_{n+1} 是终止状态时，V=0 是 bootstrap 的自然选择
        last_value: float = 0.0
        if last_obs is not None:
            # ---- 将 last_obs 转换为张量 ----
            last_obs_tensor: torch.Tensor = (
                torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )  # shape: (1, state_dim)

            with torch.no_grad():
                _, _, last_val_tensor = self.network.get_action(last_obs_tensor)
                last_value = float(last_val_tensor.item())

        # ----------------------------------------------------------------
        # Step 2: 计算 GAE 优势估计和折扣回报
        # ----------------------------------------------------------------
        # GAE(γ, λ) 公式:
        #   δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        #   A_t^GAE = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
        #   G_t = A_t + V(s_t)  (折扣回报)
        #
        # λ=0 退化为 TD(0) (高偏差, 低方差)
        # λ=1 退化为 Monte Carlo (低偏差, 高方差)
        # λ=0.95 是常用的平衡值
        self.buffer.compute_gae(last_value=last_value)

        # ----------------------------------------------------------------
        # Step 3: 准备全量数据并标准化优势
        # ----------------------------------------------------------------
        n: int = self.buffer.size  # 有效数据量（应等于 n_steps）

        # ---- 从缓冲区取出优势做全局标准化 ----
        advantages_np: np.ndarray = self.buffer.advantages[:n]  # shape (n,)
        returns_np: np.ndarray = self.buffer.returns[:n]        # shape (n,)

        # 转换为 PyTorch 张量进行标准化
        advantages_tensor: torch.Tensor = torch.as_tensor(
            advantages_np, dtype=torch.float32, device=self.device
        )  # shape (n,)

        # ---- 标准化优势 (Advantage Normalization) ----
        adv_mean: torch.Tensor = advantages_tensor.mean()
        adv_std: torch.Tensor = advantages_tensor.std()
        normalized_advantages: torch.Tensor = (
            advantages_tensor - adv_mean
        ) / (adv_std + 1e-8)  # shape (n,)

        # ---- 写回 buffer，确保 minibatch 读取到标准化后的优势 ----
        self.buffer.advantages[:n] = normalized_advantages.cpu().numpy()

        returns: torch.Tensor = torch.as_tensor(
            returns_np, dtype=torch.float32, device=self.device
        )  # shape (n,)

        # ----------------------------------------------------------------
        # Step 4: PPO 多轮 Minibatch 更新
        # ----------------------------------------------------------------
        # 每个 rollout 数据被重复使用 n_epochs 轮
        # 每轮将数据重新 shuffle 后划分为 minibatch 进行 SGD
        clip_fractions_list: List[float] = []
        approx_kl_list: List[float] = []
        policy_loss_list: List[float] = []
        value_loss_list: List[float] = []
        entropy_loss_list: List[float] = []

        for epoch in range(self.n_epochs):
            # ---- 每 epoch 重新获取 shuffled minibatches ----
            for batch in self.buffer.get_minibatches(
                batch_size=self.batch_size, shuffle=True
            ):
                # ---- 5b: 解包 minibatch ----
                (
                    states,        # (batch_size, state_dim)
                    actions,       # (batch_size,)
                    mb_returns,    # (batch_size,)
                    mb_advantages, # (batch_size,)
                    mb_old_log_probs,  # (batch_size,)
                    mb_old_values,     # (batch_size,)
                ) = batch

                # ---- 5c: 前向传播 ----
                # logits: (batch_size, n_actions) — 未经 softmax 的原始分数
                # values: (batch_size, 1) — Critic 输出的状态价值
                logits: torch.Tensor
                values: torch.Tensor
                logits, values = self.network.forward(states)
                values = values.squeeze(-1)  # (batch_size, 1) -> (batch_size,)

                # ---- 5d: 计算新策略的 log_probs ----
                # 使用 log_softmax 替代 log(softmax(x))，数值稳定性更好
                new_log_probs_all: torch.Tensor = torch.log_softmax(
                    logits, dim=-1
                )  # shape: (batch_size, n_actions)

                # 通过 gather 收集所选动作对应的 log_prob
                new_log_probs: torch.Tensor = new_log_probs_all.gather(
                    1, actions.unsqueeze(-1)
                ).squeeze(-1)  # shape: (batch_size,)

                # ---- 5e: 计算策略分布的熵 ----
                # H(π(·|s)) = -Σ_a π(a|s) * log π(a|s)
                # 熵越大 → 策略越随机 → 探索越多
                probs: torch.Tensor = torch.softmax(
                    logits, dim=-1
                )  # shape: (batch_size, n_actions)
                entropy: torch.Tensor = -(probs * new_log_probs_all).sum(
                    dim=-1
                )  # shape: (batch_size,)

                # ---- 5f: 重要性采样比率 ----
                # r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
                # 使用 exp(log_new - log_old) 保证数值稳定
                ratio: torch.Tensor = torch.exp(
                    new_log_probs - mb_old_log_probs
                )  # shape: (batch_size,)

                # ---- 5g: Clipped Surrogate Objective ----
                # L^CLIP(θ) = E_t[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]
                #   - surr1: 未裁剪的目标函数（标准重要性采样）
                #   - surr2: 裁剪后的目标函数（限制更新幅度）
                #   - 取最小值: 如果 surr1 > surr2 表明策略更新过大，使用裁剪值
                surr1: torch.Tensor = ratio * mb_advantages         # (batch_size,)
                surr2: torch.Tensor = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * mb_advantages
                )  # shape: (batch_size,)
                policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()
                # 取负号: 梯度上升（最大化期望回报）转为梯度下降（最小化负损失）

                # ---- 5h: Clipped Value Loss ----
                # 与策略裁剪类似，防止价值函数更新过大
                # L^VF(θ) = 0.5 * E_t[ max( (V_θ - G_t)^2, (V_clipped - G_t)^2 ) ]
                #   其中 V_clipped = V_old + clip(V_θ - V_old, -ε, +ε)
                #
                # 裁剪动机: 当价值更新过于激进时，使用裁剪的版本
                # 这样可以避免价值函数的单次更新破坏之前的学习成果
                value_pred_clipped: torch.Tensor = mb_old_values + torch.clamp(
                    values - mb_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon,
                )  # shape: (batch_size,)

                # 未裁剪的 MSE 损失
                value_loss_unclipped: torch.Tensor = (
                    values - mb_returns
                ).pow(2)  # shape: (batch_size,)

                # 裁剪后的 MSE 损失
                value_loss_clipped: torch.Tensor = (
                    value_pred_clipped - mb_returns
                ).pow(2)  # shape: (batch_size,)

                # 取两者最大值（保守估计，只选择更差的拟合）
                value_loss: torch.Tensor = (
                    0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                )

                # ---- 5i: 熵正则项 ----
                # H(π) 鼓励策略保持随机性，防止过早收敛到确定性策略
                # 在总损失中以 +c2 * (-entropy) 形式出现（见下一节）
                entropy_loss: torch.Tensor = -entropy.mean()

                # ---- 5j: 总损失 ----
                # L_t(θ) = L^CLIP(θ) + c1 * L^VF(θ) + c2 * H(π)
                #   - L^CLIP: 策略损失（最大化期望回报）
                #   - c1 * L^VF: 价值损失（最小化值函数误差）
                #   - c2 * (-H(π)): 熵奖励（鼓励探索）
                total_loss: torch.Tensor = (
                    policy_loss
                    + self.c1 * value_loss
                    + self.c2 * entropy_loss
                )

                # ---- 5k: 反向传播 ----
                # 清除上一次的梯度
                self.optimizer.zero_grad()

                # 反向传播计算梯度
                total_loss.backward()

                # ---- 5l: 梯度裁剪 ----
                # 将所有参数的梯度 L2 范数限制在 max_grad_norm 以内
                # 防止梯度爆炸，尤其在 RNN 或深层网络中
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm,
                )

                # 执行一步 Adam 更新
                self.optimizer.step()

                # ---- 5m: 指标记录 ----
                # 近似 KL 散度 (Kullback-Leibler Divergence)
                # 衡量新旧策略之间的差异程度
                # 方法 1: 二阶近似 KL(q||p) ≈ 0.5 * mean((log q - log p)^2)
                approx_kl: torch.Tensor = (
                    (new_log_probs - mb_old_log_probs).pow(2) / 2.0
                ).mean()

                # 方法 2 (可选): 无偏估计
                # approx_kl = (ratio - 1.0 - ratio.log()).mean()

                # 裁剪分数: 被裁剪到 [1-ε, 1+ε] 范围外的比率占比
                # 衡量当前 minibatch 中有多少比例的更新被限制
                with torch.no_grad():
                    clip_mask: torch.Tensor = (
                        (ratio < 1.0 - self.clip_epsilon)
                        | (ratio > 1.0 + self.clip_epsilon)
                    ).float()  # shape: (batch_size,)
                    clip_fraction: torch.Tensor = clip_mask.mean()

                # ---- 记录 minibatch 级指标 ----
                policy_loss_list.append(float(policy_loss.item()))
                value_loss_list.append(float(value_loss.item()))
                entropy_loss_list.append(float(entropy_loss.item()))
                approx_kl_list.append(float(approx_kl.item()))
                clip_fractions_list.append(float(clip_fraction.item()))

                # ---- KL 散度早停 (标准 PPO: 仅停止当前 epoch) ----
                # 如果 minibatch 的近似 KL 超过阈值，提前结束当前 epoch
                # 注意: 只 break 内层 minibatch 循环，下一 epoch 重新 shuffle 后继续
                if (
                    self.target_kl is not None
                    and float(approx_kl.item()) > self.target_kl
                ):
                    break

        # ----------------------------------------------------------------
        # Step 6: 计算解释方差 (Explained Variance)
        # ----------------------------------------------------------------
        # EV = 1 - Var(returns - values) / Var(returns)
        #
        # 解释方差衡量价值函数对回报的拟合程度:
        #   - EV ≈ 1:  价值函数完美拟合回报
        #   - EV ≈ 0:  价值函数与回报无关（等价于常数预测）
        #   - EV < 0:  价值函数比常数预测更差（欠拟合）
        #
        # 这里使用收集数据时的旧价值（old_values），而非更新后的价值
        # 这样反映的是更新前的拟合质量
        with torch.no_grad():
            old_values_tensor: torch.Tensor = torch.as_tensor(
                self.buffer.values[:n],
                dtype=torch.float32,
                device=self.device,
            )  # shape (n,)
            returns_all_tensor: torch.Tensor = torch.as_tensor(
                self.buffer.returns[:n],
                dtype=torch.float32,
                device=self.device,
            )  # shape (n,)

            ev_numerator: torch.Tensor = torch.var(
                returns_all_tensor - old_values_tensor
            )
            ev_denominator: torch.Tensor = torch.var(returns_all_tensor) + 1e-8
            explained_variance: torch.Tensor = 1.0 - ev_numerator / ev_denominator

        # ----------------------------------------------------------------
        # Step 7: 汇总指标
        # ----------------------------------------------------------------
        n_updates: int = len(policy_loss_list)  # 实际执行的 optimizer step 数
        metrics: Dict[str, float] = {
            "policy_loss": (
                float(np.mean(policy_loss_list)) if policy_loss_list else 0.0
            ),
            "value_loss": (
                float(np.mean(value_loss_list)) if value_loss_list else 0.0
            ),
            "entropy_loss": (
                float(np.mean(entropy_loss_list)) if entropy_loss_list else 0.0
            ),
            "approx_kl": (
                float(np.mean(approx_kl_list)) if approx_kl_list else 0.0
            ),
            "clip_fraction": (
                float(np.mean(clip_fractions_list)) if clip_fractions_list else 0.0
            ),
            "explained_variance": float(explained_variance.item()),
            "n_updates": float(n_updates),
        }

        # 记录到历史
        for key, value in metrics.items():
            if key in self._metrics_history:
                self._metrics_history[key].append(value)

        # ----------------------------------------------------------------
        # Step 8: 清空缓冲区，为下一次 rollout 做准备
        # ----------------------------------------------------------------
        self.buffer.reset()
        self._last_state = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None

        return metrics

    # ======================================================================
    # save() — 保存模型和优化器状态
    # ======================================================================
    def save(self, path: str) -> None:
        """
        保存 PPO 智能体的网络参数和优化器状态。

        保存内容包括:
            - 网络权重 (network_state_dict)
            - 优化器状态 (optimizer_state_dict)
            - 超参数配置 (config) — 用于加载时校验

        Args:
            path: 保存路径，通常以 .pt 或 .pth 结尾
        """
        # 构建保存字典
        checkpoint: Dict[str, object] = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "c1": self.c1,
                "c2": self.c2,
                "lr": self.lr,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "max_grad_norm": self.max_grad_norm,
                "target_kl": self.target_kl,
                "device": str(self.device),
            },
        }

        # 保存到磁盘
        torch.save(checkpoint, path)

    # ======================================================================
    # load() — 加载模型和优化器状态
    # ======================================================================
    def load(self, path: str) -> None:
        """
        从文件加载 PPO 智能体的网络参数和优化器状态。

        Args:
            path: 之前通过 save() 保存的文件路径
        """
        # 加载检查点
        checkpoint: Dict[str, object] = torch.load(
            path, map_location=self.device, weights_only=False
        )

        # 恢复网络权重
        network_state_dict: Dict[str, torch.Tensor] = checkpoint["network_state_dict"]
        self.network.load_state_dict(network_state_dict)

        # 恢复优化器状态
        optimizer_state_dict: Dict[str, object] = checkpoint["optimizer_state_dict"]
        self.optimizer.load_state_dict(optimizer_state_dict)

    # ======================================================================
    # 辅助方法
    # ======================================================================
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """
        获取所有历史指标。

        Returns:
            键为指标名称，值为每次 update() 记录的数值列表。
        """
        return dict(self._metrics_history)

    def reset_metrics_history(self) -> None:
        """清空指标历史记录。"""
        for key in self._metrics_history:
            self._metrics_history[key] = []

    def __repr__(self) -> str:
        return (
            f"PPOAgent(state_dim={self.state_dim}, n_actions={self.n_actions}, "
            f"hidden_dims={self.hidden_dims}, device={self.device})"
        )
