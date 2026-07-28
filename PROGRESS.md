# 项目进度记录

## 状态

- 开始：2026-07-23
- 第三轮 P0 修复：2026-07-28
- 状态：核心算法正确性已基本达标，Notebook 同步收尾中

## 第三轮修复 (2026-07-28)

### 已修复
- GridWorld: 目标状态吸收 + 转移奖励 + reset()返回(obs,info)
- TabularMCAgent: First-Visit MC 正向筛选（原为反向=Last-Visit）
- Tabular TD agents: 全部添加 terminated 参数，bootstrap_mask=1-terminated
- CI: setup-uv@v5, 移除 python-version
- 12/12 tests passing

### Agent 处理中
- Notebook 06: First-Visit MC + 5-tuple 接口
- Notebook 15: 重复代码清理 + boundary 修复
- Notebook 20: KL early stop + clip fraction 修正

## 已知限制

1. 连续动作 PPO 待补充
2. 向量化 A2C 待补充
3. 周测文件待补充
4. 部分 Notebook 仍有硬编码绝对路径
