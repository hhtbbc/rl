"""
强化学习速成教程 — 核心 Python 包

本包提供所有 Notebook 共用的可复用组件：
- envs: 自定义环境（GridWorld, Bandit 等）
- agents: 算法实现（DQN, REINFORCE, A2C, PPO 等）
- networks: 神经网络模块
- buffers: 经验回放和 rollout 缓冲区
- utils: 训练工具、种子设置、日志
- visualization: 绑图和视频生成

全部组件支持 CPU 运行，GPU 可选。
"""

__version__ = "1.0.0"
__author__ = "RL Course"

# 核心工具便捷导入
from rl_course.utils.seeding import set_seed, get_device

__all__ = ["set_seed", "get_device"]
