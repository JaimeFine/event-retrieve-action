# 测试结果 CSV 说明（`run_online_comparison.py`）

本文档说明 `scripts/run_online_comparison.py` 导出的逐轮结果 CSV（例如 `formal_hard_seed25_e10_s2000.csv`）每一列的含义。

---

## 1. 文件用途

该 CSV 是 **每轮（per-episode）明细表**，每一行对应：

- 一个难度配置下
- 某个方法（如 `era_ep100` / `era_ep200` / `bc_il` / `ppo` / `acados`）
- 的某一轮 episode 测试结果

适合用于：

- 箱线图（boxplot）
- 均值±方差/置信区间图
- 不同方法逐轮分布比较

---

## 2. 列字段说明

### `seed`
本次 run 的随机种子（例如 `25`）。

### `difficulty_mode`
难度模式，通常为：
- `fixed`
- `curriculum`

### `difficulty_level`
测试难度等级：
- `easy`
- `medium`
- `hard`
- `extreme`

### `method_key`
方法键名（用于程序识别），例如：
- `era_ep100`
- `era_ep200`
- `bc_il`
- `ppo`
- `acados`

### `method_name`
方法显示名（由适配器返回），例如：
- `era`
- `bc_il`
- `ppo`
- `acados_grid`

### `episode_index`
第几轮 episode（从 `0` 开始计数）。

### `episode_seed`
该轮实际 seed（通常是 `seed + episode_index`）。

### `effective_steps`
该轮实际有效步数（可能小于 `steps` 上限，因提前结束）。

### `success_rate`
该轮成功率（当前脚本中的步级定义）。

### `collision_rate`
该轮碰撞比例（越低越好）。

### `warning_rate`
该轮告警比例（越低越好）。

### `final_distance`
该轮结束时到目标的距离（越低越好）。

### `phys_loss`
该轮物理相关损失统计。

### `perf_loss`
该轮性能项损失统计。

### `intruder_loss`
该轮入侵者相关损失统计。

### `difficulty_name`
该轮环境记录的难度标签（通常与 `difficulty_level` 一致）。

### `difficulty_scalar`
该轮难度标量值（数值化难度强度）。

### `avg_reaction_time_ms`
该轮平均反应时间（毫秒），即动作决策耗时均值。

### `num_reaction_samples`
该轮反应时间采样数量（一般与该轮动作决策次数相关）。

---

## 3. 一行数据如何理解

一行数据可理解为：

> 在某个 `seed` 和某个 `difficulty_level` 下，
> 某个方法在第 `episode_index` 轮的完整表现（安全/完成度/时延）。

---

## 4. 论文绘图建议

建议至少画三类图：

1. **安全性**：`collision_rate`, `warning_rate`
2. **任务质量**：`success_rate`, `final_distance`
3. **实时性**：`avg_reaction_time_ms`

并按 `difficulty_level` 分组，方法作为 hue/颜色。

---

## 5. 注意事项

- `method_key` 与 `method_name` 可能不同（例如 `acados` vs `acados_grid`），画图时建议统一用 `method_key` 分组。
- 当前 `success_rate` 为脚本定义的步级指标，建议结合 `final_distance` 一起解释，避免单指标误判。
- 如果做统计检验，建议增加 seed 数量（而不只单 seed）。
