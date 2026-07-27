"""
全局随机种子设置和设备选择工具

所有实验必须固定随机种子以保证可复现性。
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """
    设置所有库的随机种子。

    Args:
        seed: 随机种子值
        deterministic: 是否启用 PyTorch 确定性模式（可能降低性能）

    Returns:
        使用的种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 环境变量中的 Python hash 种子
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    return seed


def get_device(force_cpu: bool = False) -> torch.device:
    """
    获取可用设备（优先 GPU）。

    Args:
        force_cpu: 强制使用 CPU

    Returns:
        torch.device
    """
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
