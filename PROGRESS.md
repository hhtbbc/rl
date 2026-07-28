# 项目进度记录

## 状态

- 开始：2026-07-23
- 第七轮 P0 修复：2026-07-28
- 状态：18/18 tests passing, 26/26 notebooks syntax-clean

## 第七轮修复 (2026-07-28)

### P0 修复
- **A2C rollout-end bootstrap**: 非终止 rollout 末尾从最后一个 next_value 初始化 G
  （而非 0），确保 R_{T-1} = r_{T-1} + γ·V(s_T)
- **Notebook 08 最大化偏差实验**: 完全重写为 Sutton & Barto 经典 MDP
  (State A → left/right, State B → many noisy actions)。
  Q-Learning 错误偏好右（max 选出被正噪声高估的 B 动作），
  Double Q-Learning 正确学会选左。B 动作越多偏差越明显。
- **GridWorld 默认 goal**: goal_pos 默认为 None，自动计算为 (height-1, width-1)。
  添加完整参数验证（边界、start≠goal、不在障碍上、slip_prob∈[0,1] 等）。
- **Notebook 08 总结表**: 改为条件性表述，注明排序是 CliffWalking 特定现象

### P1 修复
- **PPO 指标清理**: `approx_kl` → `pre_step_approx_kl`, `clip_fraction` → `pre_step_clip_fraction`
  (标注为 optimizer.step() 前快照)；`final_full_batch_kl` 为 post-epoch 全量 KL
- **A2CAgent**: `hidden_dims` 从可变默认值 [64,64] 改为 None
- **Notebook 08**: 移除硬编码 `/workspace/data/vggt-omega/rl` 路径
- **Notebook 18**: 更新为新 PPO 指标名

### 测试体系 (18 tests)
新增 6 个精确数值测试:
- A2C n-step returns (non-terminal cutoff)
- A2C mid-rollout terminated boundary (no cross-episode contamination)
- GridWorld auto-computed default goal
- GridWorld parameter validation (goal oob, start==goal, blocked start)
- Tabular TD targets (terminated=True vs False 精确手算值)
- First-Visit MC with repeated state (验证只取首次访问)

### 已知限制

1. 连续动作 PPO 待补充
2. 向量化 A2C 待补充
3. 周测文件 / 练习题参考答案 待补充
4. TD(λ) eligibility traces / forward-backward view 待补充
5. POMDP / Deadly Triad / Baird Counterexample 待补充
6. 多种子统计 (IQM, 置信区间) 待补充
7. Notebook 快速模式待实现真正 FAST_MODE 环境变量
8. MC truncation 处理待明确任务定义
