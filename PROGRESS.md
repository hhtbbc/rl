# 项目进度记录

## 状态

- 开始：2026-07-23
- 第九轮修复：2026-07-28
- 状态：21/21 tests passing, 26/26 notebooks syntax-clean

## 第九轮修复 (2026-07-28)

### P0 修复
- **Notebook 08 最大化偏差实验**: QL/DQL 完全独立运行；参考线修正为 100%（贪婪策略）；
  随机打破 argmax 并列；State B 使用 ε-greedy 行为策略
- **Notebook 08 CliffWalking**: 修复 cell 覆盖 bug（NotebookEdit 意外删除了环境类）；
  路径修正为 path_length() = len(path)-1；off-policy 表述修正
- **Notebook 05**: 修复迭代索引超出范围（VI 只有 9 次迭代但硬编码 index 10）
- **Notebook 13 baseline 理论表**: 修正为严谨表述（V 不是严格最小方差；
  学习 V_phi 仍无偏只要 detach；准确度影响方差而非改变无偏性）
- **Notebook 08 Q-Learning**: 补充完整收敛条件

### P1 修复
- **PPO**: KL 始终计算（不依赖 target_kl 开关），target_kl 仅控制 early stopping；
  hidden_dims 改为 Optional[List[int]] = None
- **RolloutBuffer**: shuffle 使用 self.rng (不再用全局 np.random)
- **ReplayBuffer**: sample 使用 self.rng
- **A2C compute_n_step_returns**: 增加输入验证（长度一致、done/terminated 一致性、gamma 范围）
- **Notebook 08**: 移除硬编码路径
- **测试**: 新增 3 个公式级测试 (GAE exact values, A2C input validation, replay buffer empty guard)，21 tests total

### 已知限制

1. 连续动作 PPO / tanh squashing 待补充
2. 向量化 A2C/PPO 待补充
3. TD(λ) eligibility traces 待补充
4. POMDP / Deadly Triad / Baird Counterexample 待补充
5. 周测文件 / 答案分离 待补充
6. 多种子统计 (IQM, 置信区间) 待补充
7. Notebook 快速模式 (FAST_MODE) 待实现
8. Notebook 13 baseline 实验可用 known q_values 进一步提升严谨性
