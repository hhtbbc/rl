# 强化学习速成教程 — 28 天从基础到 PPO

## 课程简介

这是一个系统的、以 Jupyter Notebook 为核心的强化学习速成教程。课程覆盖从 MDP 基础到 PPO 实现的完整知识链，强调**理论推导与代码实践并重**。

**核心理念**：不跳过数学推导，不依赖黑盒库，从直觉→数学→伪代码→PyTorch 实现逐层展开。

## 适合人群

- 有 Python 和 PyTorch 基础，RL 基础较弱的学习者
- 想深入理解 RL 算法原理（而不只是调用库）的工程师
- 准备 RL 相关面试的求职者
- 需要快速上手 RL 项目的研究生

## 预期学习成果

完成本课程后，你将能够：

1. 理解 RL 核心数学原理（MDP, Bellman, MC, TD, Policy Gradient, PPO, GAE）
2. 从零实现主要算法（DQN, REINFORCE, A2C, PPO）
3. 独立阅读常见 RL 论文和开源代码
4. 修改环境、网络、奖励函数和训练参数
5. 分析训练曲线、定位不收敛原因
6. 完成小型 RL 项目
7. 具备 RL 面试所需的理论与实践基础

## 28 天学习路线

### 第一周：RL 基础与 Tabular 方法

| 天 | 内容 | Notebook | 预计时间 |
|----|------|----------|----------|
| 1 | RL 概述 + 数学准备（上） | 01, 02 | 3h |
| 2 | 数学准备（下）+ 多臂老虎机 | 02, 03 | 3h |
| 3 | MDP 形式化 | 04 | 3h |
| 4 | Bellman 方程与动态规划 | 05 | 3h |
| 5 | Monte Carlo 方法 | 06 | 3h |
| 6 | TD 学习 | 07 | 3h |
| 7 | 复习 + SARSA/Q-Learning | 08 | 4h |

### 第二周：价值函数逼近与 DQN

| 天 | 内容 | Notebook | 预计时间 |
|----|------|----------|----------|
| 8 | 函数逼近 | 09 | 2h |
| 9 | DQN 理论与实现 | 10 | 3h |
| 10 | DQN 实验分析 | 10 | 3h |
| 11 | 策略梯度推导 | 11 | 3h |
| 12 | REINFORCE 实现 | 12 | 3h |
| 13 | Baseline 与 Advantage | 13 | 3h |
| 14 | 周测复习 | — | 3h |

### 第三周：策略梯度进阶

| 天 | 内容 | Notebook | 预计时间 |
|----|------|----------|----------|
| 15 | Actor-Critic | 14 | 3h |
| 16 | A2C 从零实现 | 15 | 3h |
| 17 | Importance Sampling | 16 | 2h |
| 18 | TRPO 理论 | 17 | 3h |
| 19 | GAE 推导 | 18 | 3h |
| 20 | PPO 理论 | 19 | 3h |
| 21 | 周测复习 | — | 3h |

### 第四周：实战与总结

| 天 | 内容 | Notebook | 预计时间 |
|----|------|----------|----------|
| 22 | PPO 从零完整实现 | 20 | 4h |
| 23 | RL 调试方法论 | 21 | 3h |
| 24 | 实验设计 | 22 | 2h |
| 25-26 | 结课项目 | 23 | 8h |
| 27 | 面试复习 | 24 | 3h |
| 28 | 最终评估 | 25 | 3h |

## 环境安装

### 1. 安装依赖

```bash
cd /workspace/data/vggt-omega/rl
uv sync
```

### 2. 注册 Jupyter Kernel

```bash
uv run python -m ipykernel install --user --name rl-course --display-name "Python (RL Course)"
```

### 3. 启动 Jupyter Lab

```bash
uv run jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

### 4. VSCode 推荐插件

- Python (`ms-python.python`)
- Jupyter (`ms-toolsai.jupyter`)
- Pylance (`ms-python.vscode-pylance`)

### 5. 选择 Kernel

在 VSCode 中打开任意 Notebook，右上角选择 "Python (RL Course)"

### 6. 启动 TensorBoard

```bash
uv run tensorboard --logdir=outputs/logs --bind_all
```

### 7. 运行测试

```bash
uv run pytest tests/ -q
```

## 项目结构

```
rl/
├── README.md                    # 本文件
├── pyproject.toml               # 项目配置和依赖
├── IMPLEMENTATION_PLAN.md       # 实施计划
├── PROGRESS.md                  # 进度记录
├── notebooks/                   # Jupyter Notebook 教程 (26 个, 00-25)
│   ├── 00_course_guide.ipynb
│   ├── 01_rl_overview.ipynb
│   └── ...
├── rl_course/                   # 可复用 Python 包
│   ├── agents/                  # 算法实现 (DQN, REINFORCE, A2C, PPO)
│   ├── envs/                    # 自定义环境 (GridWorld, Bandit)
│   ├── networks/                # 神经网络 (MLP, ActorCritic)
│   ├── buffers/                 # 经验回放和 Rollout 缓冲区
│   ├── utils/                   # 工具函数 (种子, 配置, 日志)
│   └── visualization/           # 绑图和视频生成
├── exercises/                   # 练习题 (待补充)
├── solutions/                   # 参考答案 (待补充)
├── tests/                       # 单元测试 (18 个)
├── scripts/                     # 辅助脚本 (notebook 验证等)
├── outputs/                     # 输出文件
│   ├── figures/                 # 图片
│   ├── videos/                  # GIF/MP4 视频
│   ├── checkpoints/             # 模型权重
│   ├── logs/                    # TensorBoard 日志
│   └── reports/                 # 实验报告
└── .github/workflows/           # CI 配置
```

## CPU/GPU 使用说明

所有代码在 CPU 上即可运行。课程会自动检测 CUDA：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- **快速验证**（`--fast`）：跳过 5 个长训练 Notebook，2-3 分钟 CPU
- **完整验证**（无 `--fast`）：所有 26 个 Notebook，10-30 分钟 CPU
- **语法检查**（`--syntax-only`）：仅编译检查，秒级完成

## 学习顺序

按 Notebook 编号顺序学习：00 → 01 → 02 → ... → 25

每个 Notebook 的前置知识在文件头部的"本节位置"中标注。

## 常见问题

### Q: 导入 rl_course 失败？

确保在项目根目录运行，且已执行 `uv sync`。

### Q: CUDA 不可用？

正常。所有实验设计为 CPU 优先。GPU 仅用于加速。

### Q: 图片/视频在哪里？

所有输出保存在 `outputs/` 目录下：
- 图片：`outputs/figures/`
- 视频：`outputs/videos/`
- 模型：`outputs/checkpoints/`

### Q: 渲染报错？

本教程不使用 `plt.show()` 或 `render_mode="human"`。所有渲染通过 Agg 后端保存为文件。

## 参考资料

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- David Silver's RL Course (UCL)
- OpenAI Spinning Up
- DQN, TRPO, GAE, PPO 原始论文
- Gymnasium 官方文档
- PyTorch 官方文档
