# TOFU Forget05 偏差分析与优化排查清单

日期：2026-08-08  
对照：论文 Table 1 SimNPO vs 本地 2GPU `checkpoint-62`  
论文基线：`TOFU/results/20251017/`  
本地结果：`TOFU/results/20260807/2GPU_forget05_summary.txt`

---

## 1. 偏差总览（真实数据）

| 指标 | 论文 SimNPO | 本地 2GPU | 偏差 | 严重度 |
|------|-------------|-----------|------|--------|
| **FQ↑** | **0.99** | **4.45e-8** | **-0.99** | **致命** |
| MU↑ | 0.58 | 0.599 | +0.02 | 接近，正常 |
| Forget ROUGE↓ | 0.26 | 0.409 | +0.15 | 欠遗忘 |
| Forget Prob↓ | 0.03 | 0.090 | +0.06 | 欠遗忘 |
| Forget 1-Rouge↑ | 0.74 | 0.591 | -0.15 | 欠遗忘 |
| Forget 1-Prob↑ | 0.97 | 0.910 | -0.06 | 欠遗忘 |
| Retain Prob | 0.56 | 0.797 | +0.24 | 保留过强 |
| Real Authors / World | ~论文 | 偏差 ≤10% | 小 | 效用侧基本正常 |

**形态判断（重要）：**

- 论文里 GA/GradDiff 的 FQ 也是 `~1e-9`，但是 **Forget 1-Rouge=1.00（过遗忘）+ MU 崩**。
- 本地是 **FQ~1e-8 + Forget 偏弱 + MU 还行** → **同一数量级的 FQ，失败模式不同：欠遗忘导致 Truth Ratio 分布对不齐 retain95**。
- 证据：raw TR mean 本地 **2.37** vs retain95 **1.54**（KS=0.295, p=4.4e-8）。

---

## 2. 已完成排查（本轮）

| ID | 项 | 结果 | 结论 |
|----|----|------|------|
| A1 | retain 对照路径 | `retain95_llama_wd0.01/.../eval_log_aggregated.json` | ✅ 正确 |
| A2 | KS / FQ 重算 | KS=0.2950, p=4.448674e-8 与 aggregate 一致 | ✅ 评测未算错 |
| A3 | 2 卡 + eval.batch_size=8 | 满足 `interleave` 约束 | ✅ |
| B1 | 损失系数（相对官方 README） | `npo=0.1375, grad_diff=1.0, β=2.5, γ=0` | ✅ 对齐官方命令 |
| B2 | 全局 batch | `1×16×2=32`，62 steps | ✅ 对齐论文 8×4 |
| B4 | origin 权重 | `locuslab/tofu_ft_llama2-7b` 快照完整 ~13GB | ✅ 存在 |
| B8 | chat 模板 / flash-attn | INST 标签齐全；flash_attention2=true | ✅ |

**已排除：** 评测算错、retain 用错、2 卡交织坏了、全局 batch 不对、origin 文件缺失、`75042c7` 系数对调未进当前分支。

---

## 3. 新发现的高优先级嫌疑（相对旧清单的升级）

### P0 — λ 放置与论文不一致（已按原文确认）

论文式 (1) 原文：

```text
minimize  E_{D_f}[ℓ_f]  +  λ · E_{D_r}[ℓ_r]
```

**λ 乘在保留项 `ℓ_r` 上**（不是遗忘项）。Appendix I.2：SimNPO 的 λ 搜索范围 `[0.05, 0.25]`；β∈`[1.5,3.5]`，γ 默认 0；每 epoch 评估并选优。

代码实现：`loss = npo_coeff * SimNPO + grad_diff_coeff * retain_CE`

| | 配置 | 是否符合论文式 (1) |
|--|------|-------------------|
| **论文正确读法（已改入本仓库）** | `npo_coeff=1.0`, `grad_diff_coeff=λ`（forget05: 0.1375） | ✅ `L_SimNPO + λ·L_CE` |
| 旧官方 README / 本地错误跑法 | `npo_coeff=0.1375`, `grad_diff_coeff=1.0` | ❌ 把 λ 错乘在 forget 上 |

本地错误跑法现象（Retain Prob 0.80≫论文 0.56、Forget 偏弱、FQ~1e-8）与「forget 被缩小 7×、retain 满权重」一致。

> 上游 `ec7be0c`（Chongyu Fan, 2024-10-09）曾把 `npo_coeff: 0.1375` 写进 yaml/README，与论文式 (1) 冲突。**2026-08-08 已按论文改回**：`forget.yaml` / `run_simnpo.sh` / README / 复现指南 / 方法原理。下一步用 P0a 重训验证 FQ。

### P1 — 只保存最终 epoch，论文按 epoch 选优

论文：*"measured after each unlearning epoch and selected the optimal one"*  
本地：`evaluation_strategy=no`，只在结束时评估一次。  
即便系数对，最终 epoch 也可能不是 FQ 最优。

### P2 — 未做 β/λ 网格，固定单点

论文 Table 1 是网格后的最佳；我们固定 `β=2.5, npo=0.1375`。

---

## 4. 优化后排查清单（按执行顺序）

### Phase 0 — 评测与设定核对（已完成）

- [x] A1 retain95 路径
- [x] A2 FQ 重算
- [x] A3 2 卡 eval 交织
- [x] B1/B2 系数与全局 batch（相对官方命令）
- [x] B4/B8 origin + 模板

### Phase 1 — 立刻做（低成本 / 定方向）【进行中】

| ID | 动作 | 目的 | 通过标准 |
|----|------|------|----------|
| C1 | 确认失败模式=欠遗忘 | TR / Forget 指标 | raw TR 本地 > retain95；Forget 弱于论文 → 已确认 |
| **P0a** | **按改正配置重训：`npo_coeff=1.0 grad_diff_coeff=0.1375`**（配置已改；`run_index` 可沿用，save_dir 会因 `grad_diff_coeff0.1375` 自动与旧跑区分） | 验证 λ 纠正后 FQ | Forget ROUGE/Prob 明显下降，FQ 数量级上升 |
| P0b | 同配置下若显存/时间紧，可先 `num_epochs=3~5` + 每 epoch `eval_checkpoint` 粗看趋势 | 快速看 TR 是否靠近 retain95 | TR mean 向 ~1.5 靠拢 |

### Phase 2 — 对齐论文训练协议

| ID | 动作 | 目的 |
|----|------|------|
| P1a | 打开按 epoch 评估（或每 `steps_per_epoch` 存 ckpt 并 eval） | 复制「选最优 epoch」 |
| P1b | 在裁定后的 λ 放置上扫 β∈{1.5,2.5,3.5} | 对齐论文搜索 |
| P1c | 可选扫 λ∈{0.05,0.125,0.1375,0.25}（放在已裁定的那一侧） | 对齐论文 λ 网格 |

### Phase 3 — 隔离训练 vs 评测（可选）

| ID | 动作 | 目的 |
|----|------|------|
| D5 | 若有作者发布的 unlearned ckpt，只跑本地 eval | 本地 FQ 管线是否能打出高分 |
| B7 | HF `tofu_ft` vs 作者 Drive origin 对比 | 排除起点权重差异 |

### 明确不再做

- 不再怀疑 A1/A2 算错  
- 不再用 6 卡 accum5/6 当主结论  
- 不把汇总里的 Forget Truth Ratio（`min(R,1/R)`）当 FQ  
- 不在未做 P0a 前盲目加大 epoch 空转  

---

## 5. 下一步（准备执行）

**立即实验 P0a（推荐）：**

```bash
cd /usr/local/Unlearn-Simple/TOFU
source /root/autodl-tmp/env_hf.sh
# 新 run_index，避免覆盖现有 checkpoint-62
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29502 \
  forget.py --config-name=forget.yaml \
  split=forget05 beta=2.5 \
  npo_coeff=1.0 grad_diff_coeff=0.1375 \
  run_index=2
```

预期目录名将含 `grad_diff_coeff0.1375`，与现有 `grad_diff_coeff1.0` 区分。

成功信号（相对当前）：

1. Forget ROUGE → 更接近 0.26（或至少明显 < 0.41）  
2. Forget Prob → 更接近 0.03（或明显 < 0.09）  
3. FQ → 从 `1e-8` 升到 `1e-3` 以上（理想接近 0.1~0.99）  
4. MU 允许略降，但不应像 GA 一样到 0  

---

## 6. 本轮状态

| 阶段 | 状态 |
|------|------|
| 偏差量化 | ✅ 完成 |
| Phase 0 | ✅ 完成 |
| Phase 1 C1 | ✅ 欠遗忘已确认 |
| Phase 1 P0a | ⏳ 待启动训练 |
