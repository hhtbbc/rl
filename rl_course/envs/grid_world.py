"""
GridWorld 环境

可配置的网格世界强化学习环境。
遵循 Gymnasium API。observation_space/action_space 属性在访问时导入 gymnasium。
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class GridWorld:
    """
    可配置的 N×M 网格世界。

    状态空间: {0, 1, ..., N*M-1}（扁平化索引）
    动作空间: {0: 上, 1: 右, 2: 下, 3: 左}

    特性:
    - 可配置起始/目标/障碍位置
    - 支持随机转移（slippery）模式
    - 无 GUI 渲染 — 仅文本和 rgb_array 模式
    - 遵循类 Gymnasium API

    Args:
        width: 网格宽度
        height: 网格高度
        start_pos: 起始位置 (row, col)
        goal_pos: 目标位置 (row, col)
        blocked_positions: 障碍位置列表
        step_reward: 每步基础奖励（通常为负或零）
        goal_reward: 到达目标奖励
        slip_prob: 随机动作概率（0=确定性）
        seed: 随机种子
    """

    # 动作映射：上、右、下、左
    ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    ACTION_NAMES = ["↑", "→", "↓", "←"]

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        blocked_positions: Optional[List[Tuple[int, int]]] = None,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        slip_prob: float = 0.0,
        max_steps: int = 100,
        seed: int = 42,
    ):
        # --- 基础参数 ---
        if width <= 0 or height <= 0:
            raise ValueError(f"width={width}, height={height} must be > 0")
        if max_steps <= 0:
            raise ValueError(f"max_steps={max_steps} must be > 0")
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError(f"slip_prob={slip_prob} must be in [0, 1]")

        self.width = width
        self.height = height

        # --- 位置默认值 ---
        if start_pos is None:
            start_pos = (0, 0)
        if goal_pos is None:
            goal_pos = (height - 1, width - 1)

        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.blocked_positions = blocked_positions or []
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.slip_prob = slip_prob
        self.max_steps = max_steps

        # --- 位置合法性验证 ---
        def _check_pos(name: str, pos: Tuple[int, int]) -> None:
            r, c = pos
            if not (0 <= r < height and 0 <= c < width):
                raise ValueError(
                    f"{name}={pos} is outside {height}×{width} grid"
                )

        _check_pos("start_pos", start_pos)
        _check_pos("goal_pos", goal_pos)
        for bp in self.blocked_positions:
            _check_pos("blocked_position", bp)

        if start_pos == goal_pos:
            raise ValueError(
                f"start_pos={start_pos} must differ from goal_pos={goal_pos}"
            )
        if start_pos in self.blocked_positions:
            raise ValueError(f"start_pos={start_pos} must not be blocked")
        if goal_pos in self.blocked_positions:
            raise ValueError(f"goal_pos={goal_pos} must not be blocked")

        self.n_states = width * height
        self.n_actions = 4

        # 状态 ↔ 坐标转换
        self._state_to_pos = {
            r * width + c: (r, c) for r in range(height) for c in range(width)
        }
        self._pos_to_state = {
            (r, c): r * width + c for r in range(height) for c in range(width)
        }

        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """重置环境 (Gymnasium 兼容接口)。

        Args:
            seed: 可选随机种子
            options: 可选字典, 支持 "start_state" 键指定初始状态索引

        Returns:
            (observation, info) 元组
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if options and "start_state" in options:
            start_state = int(options["start_state"])
            if start_state not in self._state_to_pos:
                raise ValueError(f"Invalid start_state: {start_state}")
            self.agent_pos = list(self._state_to_pos[start_state])
        else:
            self.agent_pos = list(self.start_pos)

        self.steps_taken = 0
        return self._pos_to_state[tuple(self.agent_pos)], {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        执行一个动作（Gymnasium 5 元组接口）。

        Args:
            action: 0=上, 1=右, 2=下, 3=左

        Returns:
            (observation, reward, terminated, truncated, info)

            - terminated: 智能体到达目标（自然终止）
            - truncated: 达到 max_steps 但未到达目标（时间截断）
            - 两者的区别对 GAE bootstrap 至关重要：
              terminated=True → 未来价值为 0
              truncated=True  → 仍需从最终状态 bootstrap
        """
        if self.slip_prob > 0 and self.rng.rand() < self.slip_prob:
            # 随机选择一个动作（slippery 转移）
            action = self.rng.randint(0, self.n_actions)

        dr, dc = self.ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # 边界检查：撞墙则留在原地
        if 0 <= new_r < self.height and 0 <= new_c < self.width:
            new_pos = (new_r, new_c)
            # 障碍检查
            if new_pos not in self.blocked_positions:
                self.agent_pos = [new_r, new_c]

        self.steps_taken += 1
        state = self._pos_to_state[tuple(self.agent_pos)]

        # 区分两种终止方式
        terminated = tuple(self.agent_pos) == self.goal_pos
        truncated = (self.steps_taken >= self.max_steps) and not terminated

        reward = self.goal_reward if terminated else self.step_reward

        info = {
            "steps": self.steps_taken,
            "pos": tuple(self.agent_pos),
            "terminated": terminated,
            "truncated": truncated,
        }

        return state, reward, terminated, truncated, info

    def get_available_actions(self) -> List[int]:
        """返回当前状态下合法的动作列表"""
        r, c = self.agent_pos
        actions = []
        for a, (dr, dc) in enumerate(self.ACTION_DELTAS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if (nr, nc) not in self.blocked_positions:
                    actions.append(a)
        return actions

    def render(self, mode: str = "ansi") -> Optional[str]:
        """
        渲染当前状态。

        Args:
            mode: "ansi" — 文本网格
                  "rgb_array" — numpy 数组 (height×10, width×10, 3)

        Returns:
            ansi 模式返回字符串，rgb_array 模式返回 numpy 数组
        """
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _render_ansi(self) -> str:
        """文本渲染"""
        lines = []
        for r in range(self.height):
            row_chars = []
            for c in range(self.width):
                pos = (r, c)
                if pos == tuple(self.agent_pos):
                    row_chars.append("A")
                elif pos == self.goal_pos:
                    row_chars.append("G")
                elif pos in self.blocked_positions:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)

    def _render_rgb_array(self) -> np.ndarray:
        """
        使用 matplotlib 渲染为 RGB 数组（无头服务器兼容）。

        返回 shape (cell_size*height, cell_size*width, 3) 的 uint8 数组。
        """
        cell_size = 50
        fig_width = self.width * cell_size / 100
        fig_height = self.height * cell_size / 100

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.axis("off")

        # 绘制网格
        for r in range(self.height):
            for c in range(self.width):
                pos = (r, c)
                color = "white"
                edge_color = "gray"

                if pos == tuple(self.agent_pos):
                    color = "#4CAF50"  # 绿色：智能体
                elif pos == self.goal_pos:
                    color = "#FFC107"  # 黄色：目标
                elif pos in self.blocked_positions:
                    color = "#333333"  # 深灰：障碍
                elif pos == self.start_pos:
                    color = "#E3F2FD"  # 浅蓝：起点

                rect = Rectangle(
                    (c, self.height - 1 - r),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=0.5,
                )
                ax.add_patch(rect)

        fig.canvas.draw()
        # 将画布转为 numpy 数组
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    def get_transition_matrix(self) -> np.ndarray:
        """
        计算确定性版本的转移矩阵。

        Returns:
            P: shape (n_states, n_actions, n_states)
               P[s, a, s'] = 1 当动作 a 从状态 s 必然转移到 s'
        """
        goal_state = self._pos_to_state[self.goal_pos]
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            # 目标状态是吸收状态：任何动作都留在原地
            if s == goal_state:
                P[s, :, s] = 1.0
                continue

            r, c = self._state_to_pos[s]
            for a in range(self.n_actions):
                dr, dc = self.ACTION_DELTAS[a]
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    new_pos = (nr, nc)
                    if new_pos not in self.blocked_positions:
                        ns = self._pos_to_state[new_pos]
                    else:
                        ns = s
                else:
                    ns = s  # 撞墙，留在原地
                P[s, a, ns] = 1.0
        return P

    def get_reward_matrix(self) -> np.ndarray:
        """
        计算奖励矩阵 (基于转移的期望奖励)。

        R[s, a] = Σ_{s'} P[s,a,s'] * r(s,a,s')
        其中 r(s,a,s') = goal_reward if s'=goal else step_reward

        对于确定性转移: R[s,a] = goal_reward (若动作 a 导致进入目标) 否则 step_reward
        目标状态本身 R[goal,:] = 0 (吸收/终止后无奖励)
        """
        goal_state = self._pos_to_state[self.goal_pos]
        P = self.get_transition_matrix()
        R = np.full((self.n_states, self.n_actions), self.step_reward)
        # 计算每个 state-action 的期望奖励: 只有转移到目标才给 goal_reward
        for s in range(self.n_states):
            if s == goal_state:
                R[s, :] = 0.0  # 吸收态，无后续奖励
                continue
            for a in range(self.n_actions):
                if P[s, a, goal_state] > 0:
                    # 期望奖励 = P(到目标)*goal + P(不到目标)*step
                    R[s, a] = (
                        P[s, a, goal_state] * self.goal_reward
                        + (1 - P[s, a, goal_state]) * self.step_reward
                    )
        return R

    # === Gymnasium 兼容性方法 ===
    @property
    def observation_space(self):
        """返回类 Gymnasium spaces.Discrete"""
        from gymnasium.spaces import Discrete
        return Discrete(self.n_states)

    @property
    def action_space(self):
        """返回类 Gymnasium spaces.Discrete"""
        from gymnasium.spaces import Discrete
        return Discrete(self.n_actions)


class StochasticGridWorld(GridWorld):
    """
    带随机转移的 GridWorld。

    与确定性版本的区别：
    - slip_prob > 0 时，有一定概率执行随机动作
    """

    def get_transition_matrix(self) -> np.ndarray:
        """包含 slip 概率的转移矩阵。

        转移模型与 step() 一致:
        - 以概率 slip_prob 在所有 n_actions 中均匀随机选择
        - 以概率 1-slip_prob 执行指定动作
        """
        if self.slip_prob == 0:
            return super().get_transition_matrix()

        nA = self.n_actions
        P_det = super().get_transition_matrix()  # 确定性转移 (只计算一次)
        P = np.zeros((self.n_states, nA, self.n_states))

        for s in range(self.n_states):
            for a in range(nA):
                # 指定动作: 1 - p + p/nA
                P[s, a] = (1 - self.slip_prob) * P_det[s, a]
                # 所有动作 (包括指定动作): 各 p/nA
                for oa in range(nA):
                    P[s, a] += (self.slip_prob / nA) * P_det[s, oa]
        return P
