"""
训练指标跟踪器

用于在训练过程中记录和统计 episode 回报、损失等指标。
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


class MetricTracker:
    """
    轻量级训练指标跟踪器。

    使用方式:
        tracker = MetricTracker()
        tracker.add("episode_return", 100)
        tracker.add("actor_loss", 0.5)
        print(tracker.recent_mean("episode_return", window=10))
    """

    def __init__(self):
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._steps: Dict[str, List[int]] = defaultdict(list)

    def add(self, name: str, value: float, step: Optional[int] = None) -> None:
        """记录一个标量值"""
        self._history[name].append(value)
        if step is not None:
            self._steps[name].append(step)

    def get(self, name: str) -> List[float]:
        """获取某个指标的全部历史"""
        return self._history.get(name, [])

    def recent_mean(self, name: str, window: int = 10) -> float:
        """计算最近 window 个值的均值"""
        values = self._history.get(name, [])
        if not values:
            return 0.0
        return float(np.mean(values[-window:]))

    def recent_std(self, name: str, window: int = 10) -> float:
        """计算最近 window 个值的标准差"""
        values = self._history.get(name, [])
        if not values:
            return 0.0
        return float(np.std(values[-window:]))

    def max(self, name: str) -> float:
        """获取最大值"""
        values = self._history.get(name, [])
        if not values:
            return 0.0
        return float(np.max(values))

    def summary(self, window: int = 10) -> Dict[str, float]:
        """获取所有指标的最近均值摘要"""
        return {name: self.recent_mean(name, window) for name in self._history}

    def clear(self) -> None:
        """清空所有记录"""
        self._history.clear()
        self._steps.clear()

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"MetricTracker({', '.join(self._history.keys())})"
