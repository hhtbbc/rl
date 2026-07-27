"""
智能体基类

所有强化学习算法实现的抽象基类。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch


class BaseAgent(ABC):
    """
    强化学习智能体抽象基类。

    所有算法智能体必须实现:
    - act(obs, train): 根据观测选择动作
    - update(*args, **kwargs): 执行一次参数更新

    可选实现:
    - save(path): 保存模型
    - load(path): 加载模型
    - evaluate(env, n_episodes): 评估智能体
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            from rl_course.utils.seeding import set_seed
            set_seed(seed)

    @abstractmethod
    def act(self, obs: np.ndarray, train: bool = True) -> int:
        """
        根据观测选择动作。

        Args:
            obs: 观测/状态向量
            train: True 时使用探索策略，False 时使用确定性策略

        Returns:
            动作索引
        """
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        执行一次参数更新（学习）。

        Returns:
            字典形式的损失和指标，如 {"actor_loss": 0.5, "critic_loss": 0.3}
        """
        ...

    def save(self, path: str) -> None:
        """保存模型到文件"""
        raise NotImplementedError("save() not implemented for this agent")

    def load(self, path: str) -> None:
        """从文件加载模型"""
        raise NotImplementedError("load() not implemented for this agent")

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        max_steps: int = 500,
        render: bool = False,
    ) -> Tuple[float, float]:
        """
        评估智能体（无探索噪声）。

        Args:
            env: 环境
            n_episodes: 评估的 episode 数量
            max_steps: 每 episode 最大步数
            render: 是否渲染（仅用于视频录制）

        Returns:
            (mean_return, std_return)
        """
        episode_returns = []

        for ep in range(n_episodes):
            try:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
            except Exception:
                obs = env.reset()

            total_reward = 0.0

            for _ in range(max_steps):
                action = self.act(np.array(obs), train=False)  # 确定性模式
                result = env.step(action)

                if len(result) == 5:
                    obs, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    obs, reward, done, *_ = result

                total_reward += reward

                if done:
                    break

            episode_returns.append(total_reward)

        return float(np.mean(episode_returns)), float(np.std(episode_returns))
