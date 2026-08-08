# 本地 2GPU forget05 vs 论文 Table 1 SimNPO（更新版）

## 数值对照（仓库口径）

| Metric | Paper SimNPO | Local 2GPU | delta | 解读 |
|--------|-------------:|-----------:|------:|------|
| Model Utility↑ | 0.58 | 0.599 | +0.019 | 接近 |
| Forget Quality↑ | 0.99 | 4.45e-8 | -0.99 | 致命失败 |
| Forget ROUGE↓ | 0.26 | 0.409 | +0.149 | 欠遗忘 |
| Forget Probability↓ | 0.03 | 0.090 | +0.060 | 欠遗忘 |
| Forget Truth Ratio | 0.69 | 0.541 | -0.149 | 汇总口径；FQ 看 raw R 分布 |
| Retain ROUGE | 0.54 | 0.582 | +0.042 | |
| Retain Probability | 0.56 | 0.797 | +0.237 | 保留过强 |
| Real Authors ROUGE | 0.90 | 0.857 | -0.043 | |
| Real World ROUGE | 0.90 | 0.880 | -0.020 | |

## 论文口径 Forget（1-Rouge / 1-Prob）

| | Paper | Local |
|--|------:|------:|
| 1-Rouge↑ | 0.74 | 0.591 |
| 1-Prob↑ | 0.97 | 0.910 |

## FQ 失败模式

- 非 GA 式「过遗忘崩效用」：本地 MU 仍 ~0.60，Forget 1-Rouge 仅 0.59（论文 SimNPO 0.74）。
- 是「欠遗忘 → raw Truth Ratio 分布右偏 → KS 拒绝」：
  - unlearn TR mean = 2.37
  - retain95 TR mean = 1.54
  - KS = 0.295, p = 4.45e-8

## 根因判断（按论文原文已确认 λ 位置；配置已改正）

论文式 (1)：`ℓ_f + λ·ℓ_r`，**λ 乘在保留集上**，λ∈[0.05,0.25]。  
旧跑法（官方 README）：`npo_coeff=0.1375, grad_diff=1.0` → λ 错乘在 forget 上。  
**已改正为**：`npo_coeff=1.0, grad_diff_coeff=0.1375`（见 `forget.yaml` / `run_simnpo.sh`）。

下一步：P0a 按新配置重训验证。

详见：`排查清单_forget05.md`
