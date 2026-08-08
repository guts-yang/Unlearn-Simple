# 论文基线指标（MUSE）

来源：*Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning*  
本地 PDF：`docs/Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning.pdf`  
对应表格：**Table 2**

## 文件说明

| 文件 | 内容 |
|------|------|
| `table2_muse_news_books.csv` | Table 2 原始数值（News + Books） |
| `table2_muse_news_books.txt` | 同上，可读文本版 |
| `table2_muse_news_books.png` | Table 2 截图 |
| `table2_muse_simnpo_as_summary.txt` | SimNPO（及 Retrain）→ MUSE 汇总字段名 |

## 设定

| 子任务 | 模型 |
|--------|------|
| MUSE News | LLaMA2-7B |
| MUSE Books | ICLM-7B |

| 指标 | 方向 | 含义 |
|------|------|------|
| VerbMem Df | ↓ | Forget 集逐字记忆 |
| KnowMem Df | ↓ | Forget 集知识记忆 |
| PrivLeak | → 0 | 隐私泄漏（Retrain = 0） |
| KnowMem Dr | ↑ | Retain 集知识记忆 |

## 论文 SimNPO 目标

- **News**：VerbMem 2.34，KnowMem_f 44.84，PrivLeak 72.93，KnowMem_r 39.65  
- **Books**：VerbMem 0.00，KnowMem_f 0.00，PrivLeak -19.82，KnowMem_r 48.27  

TOFU Table 1 / A3 基线见：`TOFU/results/20251017/`。
