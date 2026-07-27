"""
无头服务器兼容的可视化工具

所有绑图函数使用 matplotlib Agg 后端，保存为文件。
支持在 Notebook 中通过 IPython.display 显示图片/GIF/视频。

注意：本模块不应调用 plt.show()，仅使用 plt.savefig() 和 plt.close()。
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 强制使用 Agg 后端（无头服务器）
import matplotlib.pyplot as plt


# === 全局样式配置 ===
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)

# 确保输出目录存在
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _ensure_dir(filepath: str) -> None:
    """确保文件目录存在"""
    d = os.path.dirname(filepath)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_learning_curve(
    data: Union[List[float], Dict[str, List[float]]],
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Return",
    window: int = 10,
    filepath: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    color: Optional[str] = None,
) -> plt.Figure:
    """
    绑制学习曲线（含滑动平均）。

    Args:
        data: 列表或 {label: [values]} 字典
        title: 绑图标题
        xlabel: x 轴标签
        ylabel: y 轴标签
        window: 滑动平均窗口大小
        filepath: 保存路径（None 则自动生成）
        figsize: 图像大小
        color: 曲线颜色

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, list):
        data = {"": data}

    for label, values in data.items():
        ax.plot(values, alpha=0.3, linewidth=0.5, label=f"{label} (raw)" if label else "Raw")

        if len(values) > window:
            # 滑动平均
            smoothed = np.convolve(
                values, np.ones(window) / window, mode="valid"
            )
            ax.plot(
                range(window - 1, len(values)),
                smoothed,
                linewidth=1.5,
                color=color,
                label=f"{label} (smoothed)" if label else "Smoothed",
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, "learning_curve.png")
    _ensure_dir(filepath)
    fig.savefig(filepath)
    plt.close(fig)

    return fig


def plot_value_heatmap(
    values: np.ndarray,
    grid_shape: Tuple[int, int],
    title: str = "State Value Function",
    filepath: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    annotate: bool = True,
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    绑制状态价值热力图。

    Args:
        values: 状态价值数组，shape (n_states,)
        grid_shape: 网格形状 (height, width)
        title: 标题
        filepath: 保存路径
        figsize: 图像大小
        annotate: 是否标注数值
        cmap: 色彩映射

    Returns:
        matplotlib Figure 对象
    """
    h, w = grid_shape
    v_grid = values.reshape(h, w)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(v_grid, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Value")

    if annotate:
        for i in range(h):
            for j in range(w):
                ax.text(j, i, f"{v_grid[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, "value_heatmap.png")
    _ensure_dir(filepath)
    fig.savefig(filepath)
    plt.close(fig)

    return fig


def plot_policy_grid(
    policy: np.ndarray,
    grid_shape: Tuple[int, int],
    n_actions: int = 4,
    title: str = "Optimal Policy",
    filepath: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    action_symbols: Optional[List[str]] = None,
) -> plt.Figure:
    """
    绑制网格世界最优策略箭头图。

    Args:
        policy: 策略数组，shape (n_states,) — 每状态的最优动作索引
                或 shape (n_states, n_actions) — 动作概率分布
        grid_shape: 网格形状 (height, width)
        n_actions: 动作数量
        title: 标题
        filepath: 保存路径
        figsize: 图像大小
        action_symbols: 动作符号列表

    Returns:
        matplotlib Figure 对象
    """
    if action_symbols is None:
        action_symbols = ["↑", "→", "↓", "←"]

    h, w = grid_shape

    # 转换为每状态的动作索引
    if policy.ndim == 2:
        best_actions = np.argmax(policy, axis=1)
    else:
        best_actions = policy

    best_actions_grid = best_actions.reshape(h, w)

    # 箭头方向：上(-1,0), 右(0,1), 下(1,0), 左(0,-1)
    arrows = [(-0.4, 0), (0, 0.4), (0.4, 0), (0, -0.4)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    for r in range(h):
        for c in range(w):
            a = best_actions_grid[r, c]
            dy, dx = arrows[a]
            ax.annotate(
                "",
                xy=(c + dx, r + dy),
                xytext=(c - dx * 0.3, r - dy * 0.3),
                arrowprops=dict(arrowstyle="->", lw=2, color="steelblue"),
            )
            ax.text(c, r + 0.35, action_symbols[a], ha="center", fontsize=10, color="darkblue")

    ax.set_title(title)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, "policy_grid.png")
    _ensure_dir(filepath)
    fig.savefig(filepath)
    plt.close(fig)

    return fig


def plot_comparison(
    curves: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    xlabel: str = "Episode",
    ylabel: str = "Return",
    window: int = 10,
    filepath: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    绑制多个算法的对比曲线。

    Args:
        curves: {算法名: [values]} 字典
        title: 标题
        xlabel: x 轴标签
        ylabel: y 轴标签
        window: 滑动平均窗口
        filepath: 保存路径
        figsize: 图像大小

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, len(curves)))

    for (label, values), color in zip(curves.items(), colors):
        if len(values) > window:
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(values)), smoothed, label=label, color=color, linewidth=2)
        else:
            ax.plot(values, label=label, color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, "comparison.png")
    _ensure_dir(filepath)
    fig.savefig(filepath)
    plt.close(fig)

    return fig


def plot_multiple_metrics(
    metric_dict: Dict[str, List[float]],
    title: str = "Training Metrics",
    filepath: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 3 * 3),
    ncols: int = 3,
) -> plt.Figure:
    """
    在一个图中绑多个子图。

    Args:
        metric_dict: {名称: [values]} 字典
        title: 总标题
        filepath: 保存路径
        figsize: 图像大小
        ncols: 列数

    Returns:
        matplotlib Figure 对象
    """
    n = len(metric_dict)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = [axes]
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, (name, values), i in zip(axes, metric_dict.items(), range(n)):
        ax.plot(values, linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, "metrics.png")
    _ensure_dir(filepath)
    fig.savefig(filepath)
    plt.close(fig)

    return fig
