"""
多臂老虎机环境

经典探索-利用问题的标准环境。
支持 Bernoulli 和 Gaussian 奖励分布。
"""

from typing import Optional, Literal
import numpy as np


class MultiArmedBandit:
    """
    K-臂老虎机环境。

    每个臂有一个未知的期望奖励 μ_k，每次拉臂获得随机奖励。
    目标：在有限的拉臂次数内最大化总奖励。
    核心挑战：探索（收集信息）vs 利用（使用已知最佳臂）。

    Args:
        k: 臂的数量
        reward_type: "bernoulli" — 每臂固定概率给 1 奖励；
                     "gaussian" — 每臂高斯分布 N(μ_k, σ²)
        true_means: 可指定各臂真实均值，None 则随机生成
        sigma: Gaussian 奖励的标准差（仅 reward_type="gaussian" 时有效）
        seed: 随机种子
    """

    def __init__(
        self,
        k: int = 10,
        reward_type: Literal["bernoulli", "gaussian"] = "bernoulli",
        true_means: Optional[np.ndarray] = None,
        sigma: float = 1.0,
        seed: int = 42,
    ):
        self.k = k
        self.reward_type = reward_type
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

        if true_means is not None:
            assert len(true_means) == k, f"true_means length must be {k}"
            self.true_means = np.array(true_means, dtype=np.float32)
        else:
            # 随机生成真实均值
            if reward_type == "bernoulli":
                self.true_means = self.rng.uniform(0, 1, size=k).astype(np.float32)
            else:
                self.true_means = self.rng.randn(k).astype(np.float32)

        # 按均值排序后的索引（最优臂是最后一个）
        self.optimal_action = int(np.argmax(self.true_means))
        self.total_pulls = 0
        self.action_counts = np.zeros(k, dtype=np.int32)
        self.action_rewards = np.zeros(k, dtype=np.float32)

    def pull(self, action: int) -> float:
        """
        拉一个臂。

        Args:
            action: 臂的索引 (0 ≤ action < k)

        Returns:
            随机奖励
        """
        if action < 0 or action >= self.k:
            raise ValueError(f"Action {action} out of range [0, {self.k})")

        mean = self.true_means[action]

        if self.reward_type == "bernoulli":
            reward = float(self.rng.rand() < mean)
        else:  # gaussian
            reward = self.rng.normal(mean, self.sigma)

        self.total_pulls += 1
        self.action_counts[action] += 1
        self.action_rewards[action] += reward

        return reward

    def reset(self) -> None:
        """重置计数器（真实均值不变）"""
        self.total_pulls = 0
        self.action_counts = np.zeros(self.k, dtype=np.int32)
        self.action_rewards = np.zeros(self.k, dtype=np.float32)

    @property
    def regret(self) -> float:
        """
        累积 regret：最优期望奖励 × 总拉臂数 - 实际总奖励

        衡量算法的"后悔"程度，越小越好。
        """
        optimal_mean = self.true_means[self.optimal_action]
        expected_optimal = optimal_mean * self.total_pulls
        actual_total = self.action_rewards.sum()
        return float(expected_optimal - actual_total)

    @property
    def best_action_frac(self) -> float:
        """选择最优臂的比例"""
        if self.total_pulls == 0:
            return 0.0
        return float(self.action_counts[self.optimal_action] / self.total_pulls)
