# 论文基线指标（TOFU）

来源：*Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning*  
本地 PDF：`docs/Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning.pdf`

## 文件说明

| 文件 | 内容 |
|------|------|
| `table1_forget05_llama2_7b.csv` | Table 1 原始数值（论文报告口径） |
| `table1_forget05_llama2_7b.txt` | 同上，可读文本版 |
| `table1_forget05_llama2_7b.png` | Table 1 截图 |
| `table1_forget05_simnpo_as_aggregate_stat.txt` | SimNPO 一行 → 本仓库 `aggregate_stat.txt` 字段名 |
| `table_a3_forget10_llama2_7b.csv` | 附录 Table A3（Forget10） |
| `table_a3_forget10_llama2_7b.txt` | 同上，可读文本版 |
| `compare_local_2gpu_vs_paper_simnpo.txt` | 本地 2GPU forget05 vs 论文 SimNPO |

MUSE Table 2 基线见：`MUSE/results/20251017/`。

## 设定（Table 1）

- 数据划分：**TOFU Forget05**
- 模型：**LLaMA2-7B-chat**
- 主指标：**FQ**（Forget Quality）、**MU**（Model Utility）
- 结果为 **5 次随机试验**平均

## 指标口径说明

论文 Forget Set 列是 **`(1-Rouge-L)↑`**、**`(1-Prob.)↑`**（越大表示遗忘越强）。  
本仓库 `aggregate_stat.txt` 报告的是原始 **Forget ROUGE** / **Forget Probability**（越小表示遗忘越强）。

`table1_forget05_simnpo_as_aggregate_stat.txt` 中的换算：

```
Forget ROUGE       = 1 - (1-Rouge-L)
Forget Probability = 1 - (1-Prob.)
```

论文 SimNPO 在 Forget05 上的目标：**FQ = 0.99**，**MU = 0.58**。
