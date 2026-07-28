# 项目进度记录

## 状态

- 开始：2026-07-23
- 第二轮 P0 修复：2026-07-28
- 状态：P0 修复进行中，核心包已完成，Notebook 同步中

## P0 修复记录 (2026-07-28)

### P0.1 — terminated/truncated 全项目重构
- GridWorld.step() 改为 5-tuple `(obs, reward, terminated, truncated, info)`
- RolloutBuffer: 分离 `dones` (episode边界) 和 `terminated` (真正终止)
- compute_gae(): `bootstrap_mask = 1.0 - terminated`, GAE reset 用 `dones`
- PPOAgent.store(): 新增 `terminated` 参数
- A2CAgent: 分离 terminated/truncated 列表，bootstrap 决策基于 terminated
- ReplayBuffer: 新增 `terminated` 存储，sample() 返回 6-tuple
- DQNAgent/DoubleDQNAgent: TD target 用 `terminated` 而非 `dones`
- REINFORCE: 文档说明 5-tuple 接口

### P0.2 — PPOAgent 优势标准化修复
- 标准化后的优势写回 `self.buffer.advantages[:n]`
- Minibatch 现在读取到标准化后的优势

### P0.3 — Notebook 20 PPO from scratch 修复
- collect_rollout() 接受外部 state 参数（rollout 连续性）
- episode return 正确追踪（episode 完成时记录）
- GAE 使用 critic 计算 last_value 而非 0.0
- get_minibatches() 在每个 epoch 内调用（重新 shuffle）
- KL 早停标注为标准 PPO 行为（仅停当前 epoch）
- 指标使用实际 optimizer step 数

### P0.4 — TRPO Notebook 17 重写
- 删除错误的 SimplifiedTRPO（KL 恒为 0）
- 新增 4 个正确的组件演示：KL、HVP、CG、Line Search
- 明确说明完整 TRPO 的复杂性和 PPO 的实用性

### P0.5 — REINFORCE baseline Critic target 修复
- Critic 拟合原始（未归一化）回报
- 优势在原始尺度计算后归一化
- Basic REINFORCE 标注 per-episode 归一化为 heuristic

### P0.6 — Debug Notebook 21 重写
- 移除 BUG 1-8 标签
- 三级分类：确定错误 / 高风险问题 / 可选稳定化技巧
- 修复错误答案（value loss 描述）
- 答案折叠在最后

### P0.7 — 理论 Notebook 修复
- MDP (04): 修正 Markov 性质表述、石头剪刀布例子、Continuing task 说明
- Policy Gradient (11): 区分 trajectory REINFORCE 与 PG Theorem、修复损坏 LaTeX
- PPO Theory (19): 修正 clipping 描述、指标范围改为经验参考、PPO 不是严格信任域

## 测试状态

- 12/12 pytest 通过
- CUDA 警告：驱动 12.9 vs PyTorch 12.8，回退 CPU，不影响使用

## 已知限制

1. TRPO 提供组件演示而非完整可训练实现
2. 连续动作 PPO 待补充
3. Notebook 快速模式执行脚本待补充 (P1.3)
4. uv.lock 待提交 (P1.1)
5. 周测文件待补充 (P2.1)
