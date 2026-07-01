# 实验结果整理

结果目录位于 `gg2/output2`，核心表格与评估文件位于 `gg2/output2/verify_logs`。

## 1. 消融实验总表

来源文件：

- `output2/verify_logs/ablation_table_iter30000_20260505_165918.csv`
- `output2/verify_logs/ablation_table_iter30000_20260505_165918.json`

评测迭代数：`iter=30000`

| Scene | Variant | Model Dir | mIoU | mBIoU | Delta mIoU vs Baseline | Delta mBIoU vs Baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| figurines | baseline | `output2/figurines_baseline` | 0.5614 | 0.5448 | +0.0000 | +0.0000 |
| figurines | aniso_only | `output2/figurines_aniso_only` | 0.5713 | 0.5459 | +0.0099 | +0.0011 |
| figurines | full | `output2/figurines_full` | 0.5531 | 0.5386 | -0.0083 | -0.0062 |
| figurines | uncertain | `output2/figurines_uncertain` | 0.4929 | 0.4698 | -0.0684 | -0.0750 |
| ramen | baseline | `output2/ramen_baseline` | 0.6120 | 0.5296 | +0.0000 | +0.0000 |
| ramen | aniso_only | `output2/ramen_aniso_only` | 0.7305 | 0.6446 | +0.1185 | +0.1151 |
| ramen | full | `output2/ramen_full` | 0.7207 | 0.6300 | +0.1087 | +0.1004 |
| ramen | uncertain | `output2/ramen_uncertain` | 0.7215 | 0.6295 | +0.1095 | +0.0999 |
| room | (no models found) | `-` | - | - | - | - |
| teatime | baseline | `output2/teatime_baseline` | 0.0502 | 0.0205 | +0.0000 | +0.0000 |
| teatime | aniso_only | `output2/teatime_aniso_only` | 0.0500 | 0.0202 | -0.0001 | -0.0003 |
| teatime | full | `output2/teatime_full` | 0.0429 | 0.0130 | -0.0073 | -0.0075 |
| teatime | uncertain | `output2/teatime_uncertain` | 0.0502 | 0.0202 | +0.0000 | -0.0003 |

## 2. 消融结果简述

- `figurines`：`aniso_only` 最好，较 `baseline` 有小幅提升。
- `ramen`：提升最明显，`aniso_only` 最优，`full` 与 `uncertain` 也显著优于 `baseline`。
- `teatime`：整体性能较低，各变体差距很小，`full` 反而下降。
- `room`：当前总表中显示为 `no models found`，说明尚未形成可用评估结果。

## 3. 额外阶段性结果（figurines）

除消融表外，`figurines` 还保存了一组阶段性优化结果：

- `output2/verify_logs/eval_figurines_after_promptfix_iter30000.json`
- `output2/verify_logs/eval_figurines_after_promptfix_v2_iter30000.json`
- `output2/verify_logs/eval_figurines_after_puritymask_iter30000.json`
- `output2/verify_logs/eval_figurines_after_color_purity_iter30000.json`
- `output2/verify_logs/eval_figurines_final_iter30000.json`
- `output2/verify_logs/eval_figurines_baseline_vs_aniso_only_iter30000.json`

其中 `final` 的关键指标如下：

| Scene | Setting | mIoU | mBIoU |
| --- | --- | ---: | ---: |
| figurines | baseline | 0.5614 | 0.5448 |
| figurines | final | 0.7245 | 0.6986 |

这说明 `figurines` 在后续 prompt / purity 相关优化后，结果明显高于消融表中的几组基础变体。

## 4. 当前已有结果范围

- 已有完整消融结果的场景：`figurines`、`ramen`、`teatime`
- 已有模型目录但没有成功汇总评估结果的场景：`room`
- 已有额外阶段性优化评估的场景：`figurines`
- 训练日志位于：`output2/verify_logs/train_*.log`
- 渲染日志位于：`output2/verify_logs/render_*.log`

