# 项目进度记录

## 状态

- 开始：2026-07-23
- 第五轮 P0 修复：2026-07-28
- 状态：Notebook 回归修复完成，全部 26 个 Notebook 通过语法检查

## 第五轮修复 (2026-07-28)

### 已修复 (P0)
- 全部 Notebook：清除批量 API 迁移引入的缩进错误和重复 `done = terminated or truncated`
- Notebook 07 (RandomWalk)：完整重写为 5 元组 API，修正状态索引
- Notebook 18 (GAE)：PPO 训练循环同步新 API（store 4 参数，update 0 参数，next_value 计算）
- Notebook 23 (Capstone)：PPO 训练循环同步新 API
- Notebook 13 (Baseline)：重写无偏性数值实验（使用 score function 梯度验证）
- Notebook 08 (CliffWalking)：修正文字描述（掉崖不终止 episode）
- Notebook 20 (PPO)：修正 clip fraction 解释方向错误
- Notebook 19 (PPO Theory)：修复多行字符串语法错误
- Notebook 06 (MC)：修复训练循环缩进

### 已修复 (P1)
- PPOAgent：更新 docstring 为新 API，KL 检查移至 epoch 后，修正损失符号注释
- PPOAgent：explained_variance 使用 correction=0，KL 改用 unbiased estimator
- RolloutBuffer：增加容量溢出检查，terminated 改为 keyword-only 必填参数
- A2CAgent：修复 episode boundary 处理（dones 控制跨 episode 传播）
- TabularMCAgent：修正 docstring（ε-soft On-Policy MC Control）
- validation script：增加语法检查阶段（--syntax-only），跳过 Jupyter magic 命令
- Notebook 13：修正 variance 描述（不再声称"线性增长"）

### 测试状态
- 12/12 pytest tests passing
- 26/26 notebooks pass syntax check

## 已知限制

1. 连续动作 PPO 待补充
2. 向量化 A2C 待补充
3. 周测文件待补充
4. POMDP / TD(λ) eligibility traces / deadly triad 等进阶主题待补充
5. 多种子统计报告待补充
6. 快速 CI（--fast）跳过 5 个长训练 Notebook，执行级验证待加强
