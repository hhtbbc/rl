"""
无头服务器兼容的视频录制工具

通过 env.render("rgb_array") 收集帧，使用 imageio 生成 GIF/MP4。
"""

import os
from typing import List, Optional, Tuple, Callable
import numpy as np

OUTPUT_DIR = "outputs/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def record_episode(
    env,
    policy_fn: Callable[[int], int],
    filepath: Optional[str] = None,
    fps: int = 10,
    max_steps: int = 200,
    format: str = "gif",
    verbose: bool = False,
) -> str:
    """
    录制一个 episode。

    注意：env 必须支持 render("rgb_array") 模式。

    Args:
        env: 环境对象，需有 reset(), step(), render("rgb_array") 方法
        policy_fn: 策略函数，输入 state 输出 action
        filepath: 输出路径（None 则自动生成）
        fps: 帧率
        max_steps: 最大步数
        format: "gif" 或 "mp4"
        verbose: 是否打印步骤信息

    Returns:
        输出文件路径
    """
    import imageio

    frames: List[np.ndarray] = []

    try:
        state = env.reset()
    except TypeError:
        # 某些 env 的 reset 返回 (state, info) 元组
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

    done = False
    total_reward = 0.0
    step = 0

    while not done and step < max_steps:
        # 渲染当前帧
        try:
            frame = env.render()  # 某些 env 支持无参数调用
        except (TypeError, ValueError):
            frame = env.render(mode="rgb_array")

        if frame is not None:
            # 确保 frame 是 uint8 RGB 数组
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            frames.append(frame)

        # 选择动作并执行
        action = policy_fn(state)
        result = env.step(action)

        # 处理 gymnasium 5-tuple 返回 (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            state, reward, terminated, truncated, _ = result
            done = terminated or truncated
        else:
            state, reward, done, *_ = result

        total_reward += reward
        step += 1

        if verbose:
            print(f"  Step {step}: action={action}, reward={reward:.2f}, total={total_reward:.2f}")

    # 截最后一帧
    try:
        frame = env.render()
    except (TypeError, ValueError):
        frame = env.render(mode="rgb_array")
    if frame is not None:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        frames.append(frame)

    # 生成文件路径
    if filepath is None:
        ext = "gif" if format == "gif" else "mp4"
        filepath = os.path.join(OUTPUT_DIR, f"episode_{np.random.randint(10000)}.{ext}")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 保存为 GIF 或 MP4
    if format == "gif":
        imageio.mimsave(filepath, frames, fps=fps, loop=0)
    else:  # mp4
        imageio.mimsave(filepath, frames, fps=fps, codec="libx264")

    print(f"视频已保存: {filepath} (共{len(frames)}帧, reward={total_reward:.2f})")
    return filepath
