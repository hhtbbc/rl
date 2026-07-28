# 项目进度记录

## 状态

- 开始：2026-07-23
- 第六轮 P0 修复：2026-07-28
- 状态：P0 算法正确性问题已修复，全部 26 个 Notebook 通过语法检查，12/12 测试通过

## 第六轮修复 (2026-07-28)

### P0 修复
- **A2CAgent boundary return**：重写 n-step 回报计算。store() 每步保存 next_value，
  done 检查在 G 累积之前执行，正确区分 terminated（控制 bootstrap）和 done（控制跨 episode 传播）
- **GridWorld.reset()**：支持 `options={"start_state": s}` 参数（Gymnasium 兼容接口），
  同时支持 `seed` 关键字
- **Notebook 06 MC Control**：使用 `gw.reset(options={"start_state": start_state})` 替代
  直接修改 `agent_pos`；构建合法初始状态列表（排除目标和障碍）；统一注释为
  "First-Visit On-Policy ε-soft MC Control with random initial states"
- **Notebook 07 方差公式**：修正 MC/TD 方差公式为包含协方差项的完整形式；
  修正 V(S_{t+1}) 随机性的说明（S_{t+1} 本身在随机转移下是随机变量）；
  修正随机游走示意图为 5 非终止状态 + 2 终止边界

### P1 修复
- **PPO docstring**：target_kl 参数说明更新为 "每个 epoch 完成后全量 KL 检查"；
  early_stopped 返回 bool（非 float）；新增 `final_full_batch_kl` 指标
- **Notebook 20 PPO**：ppo_update() KL 检查从 epoch 前移至 epoch 后，
  使用 unbiased KL estimator；与包内 PPO 逻辑统一
- **Tabular agents**：所有 TD agent 的 `update()` 移除 `done` 参数（不参与 target 计算），
  `terminated` 改为 keyword-only 必填参数；新增 `self.rng = np.random.RandomState(seed)` 
  隔离每个 agent 的随机流
- **ReplayBuffer.push()**：`terminated` 改为 keyword-only 必填参数
- **Notebook 08**：训练函数更新为新 API，移除 `done` 参数传递
- **Notebook 13**：修复训练循环（清除重复 `done` 行，复用 env 并传 seed）；
  方差分析表更严谨（不再声称 "G_t:高, G_t-V:中, A:低" 的绝对关系）

### 测试体系
- 12/12 tests passing（更新为新的 keyword-only API）
- 新增 Double Q-Learning 导入测试
- 26/26 notebooks 通过语法检查

## 已知限制

1. 连续动作 PPO 待补充
2. 向量化 A2C 待补充
3. 周测文件 / 练习题参考答案 待补充
4. TD(λ) eligibility traces / forward-backward view 待补充
5. POMDP / Deadly Triad / Baird Counterexample 待补充
6. 多种子统计 (IQM, 置信区间) 待补充
7. 快速 CI（--fast）跳过 5 个长训练 Notebook，需要实现真正的 FAST_MODE 环境变量
8. Notebook 13 baseline 实验可用 known q_values 进一步提升严谨性
