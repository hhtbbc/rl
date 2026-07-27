"""
训练配置基类

所有算法配置都继承自此 dataclass，确保统一的接口。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """训练配置基类"""

    # 通用训练参数
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    total_timesteps: int = 100_000
    eval_frequency: int = 5_000
    n_eval_episodes: int = 10
    save_frequency: int = 20_000

    # 环境参数
    env_name: str = "GridWorld-v0"
    gamma: float = 0.99  # 折扣因子

    # 日志
    log_dir: str = "outputs/logs"
    checkpoint_dir: str = "outputs/checkpoints"

    # 快速模式
    fast_mode: bool = False  # 启用时减少训练步数以快速验证

    def __post_init__(self):
        if self.fast_mode:
            self.total_timesteps = min(self.total_timesteps, 5_000)
            self.eval_frequency = min(self.eval_frequency, 1_000)
