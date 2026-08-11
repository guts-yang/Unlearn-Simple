#!/usr/bin/env python3
"""Write a TOFU-style MUSE summary + update 对照表 from eval CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PAPER = {
    "news": {
        "verbmem_f": 2.34,
        "knowmem_f": 44.84,
        "privleak": 72.93,
        "knowmem_r": 39.65,
    },
    "books": {
        "verbmem_f": 0.00,
        "knowmem_f": 0.00,
        "privleak": -19.82,
        "knowmem_r": 48.27,
    },
}

RETRAIN = {
    "news": {"verbmem_f": 20.75, "knowmem_f": 33.32, "privleak": 0.00, "knowmem_r": 53.79},
    "books": {"verbmem_f": 14.30, "knowmem_f": 28.90, "privleak": 0.00, "knowmem_r": 74.50},
}

DIRS = {
    "verbmem_f": "↓",
    "knowmem_f": "↓",
    "privleak": "→0",
    "knowmem_r": "↑",
}


def fmt(x: float) -> str:
    return f"{x:.2f}"


def delta(ours: float, paper: float, key: str) -> str:
    d = ours - paper
    if key == "privleak":
        # closer to 0 is better; report signed gap to paper
        return f"{d:+.2f}"
    return f"{d:+.2f}"


def write_summary(
    corpus: str,
    row: dict,
    out_path: Path,
    *,
    ckpt: str,
    log: str,
    hparams: str,
    train_loss: str | None,
) -> None:
    paper = PAPER[corpus]
    retrain = RETRAIN[corpus]
    title = "News" if corpus == "news" else "Books"
    lines = [
        f"# MUSE {title} · SimNPO+GDR local train+eval (20260811)",
        "",
        f"Paper: Simplicity Prevails (Table 2 · MUSE {title})",
        f"Corpus: {corpus}",
        "Algo: simnpo_gdr",
        hparams,
        f"Ckpt: {ckpt}",
        f"Log train: see /root/autodl-tmp/logs/muse_{corpus}_simnpo_gdr_*.log",
        f"Log eval: {log}",
        f"CSV: {out_path.parent / f'out_{corpus}_simnpo_gdr.csv'}",
    ]
    if train_loss:
        lines.append(f"train_loss: {train_loss}")
    lines += ["", "==== Metrics (local) ===="]
    for k in ("verbmem_f", "knowmem_f", "privleak", "knowmem_r"):
        lines.append(f"{k}: {float(row[k]):.6g}")

    lines += [
        "",
        "==== 对照 (Table 2 SimNPO) ====",
        f"{'Metric':<12} | {'Dir':>3} | {'Paper':>8} | {'Retrain':>8} | {'This run':>8} | {'Δ vs paper':>10}",
        f"{'-'*12}-+-{'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}",
    ]
    for k in ("verbmem_f", "knowmem_f", "privleak", "knowmem_r"):
        ours = float(row[k])
        lines.append(
            f"{k:<12} | {DIRS[k]:>3} | {paper[k]:8.2f} | {retrain[k]:8.2f} | {ours:8.2f} | {delta(ours, paper[k], k):>10}"
        )

    lines += [
        "",
        "解读：",
        "- VerbMem/KnowMem_f：越低遗忘越强；KnowMem_r：越高效用越好；PrivLeak：越接近 0 越好（Retrain=0）。",
        "- 与论文同方法同设定对照；本机 2×A800、device_map=auto、save_total_limit=2、gradient_checkpointing。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def update_duizhao(results_root: Path, corpus: str, row: dict) -> None:
    path = results_root / "对照表.md"
    paper = PAPER[corpus]
    ours = {k: float(row[k]) for k in paper}
    title = "News" if corpus == "news" else "Books"

    # Build / refresh a simple table file covering News (and Books when present)
    news_row = None
    books_row = None
    news_csv = results_root / "20260811" / "out_news_simnpo_gdr.csv"
    books_csv = results_root / "20260811" / "out_books_simnpo_gdr.csv"
    if news_csv.exists():
        with news_csv.open() as f:
            news_row = next(csv.DictReader(f))
    if books_csv.exists():
        with books_csv.open() as f:
            books_row = next(csv.DictReader(f))
    if corpus == "news":
        news_row = {k: str(ours[k]) for k in ours} | {"name": "simnpo_gdr"}
    if corpus == "books":
        books_row = {k: str(ours[k]) for k in ours} | {"name": "simnpo_gdr"}

    def cell(r, k):
        return f"{float(r[k]):.2f}" if r and k in r else "—"

    body = f"""# MUSE · SimNPO 结果对照（Table 2）

来源：*Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning*  
本地 PDF：`docs/Simplicity Prevails Rethinking Negative Preference Optimization for LLM Unlearning.pdf`  
口径：VerbMem / KnowMem_f **↓**；PrivLeak **→0**；KnowMem_r **↑**。

## MUSE News（LLaMA2-7B）

| 来源 | VerbMem↓ | KnowMem_f↓ | PrivLeak→0 | KnowMem_r↑ |
| --- | ---: | ---: | ---: | ---: |
| **论文 Table 2 SimNPO** | {PAPER['news']['verbmem_f']:.2f} | {PAPER['news']['knowmem_f']:.2f} | {PAPER['news']['privleak']:.2f} | {PAPER['news']['knowmem_r']:.2f} |
| 论文 Retrain（参考） | {RETRAIN['news']['verbmem_f']:.2f} | {RETRAIN['news']['knowmem_f']:.2f} | {RETRAIN['news']['privleak']:.2f} | {RETRAIN['news']['knowmem_r']:.2f} |
| **20260811 本地训练+评** | {cell(news_row, 'verbmem_f')} | {cell(news_row, 'knowmem_f')} | {cell(news_row, 'privleak')} | {cell(news_row, 'knowmem_r')} |

## MUSE Books（ICLM-7B）

| 来源 | VerbMem↓ | KnowMem_f↓ | PrivLeak→0 | KnowMem_r↑ |
| --- | ---: | ---: | ---: | ---: |
| **论文 Table 2 SimNPO** | {PAPER['books']['verbmem_f']:.2f} | {PAPER['books']['knowmem_f']:.2f} | {PAPER['books']['privleak']:.2f} | {PAPER['books']['knowmem_r']:.2f} |
| 论文 Retrain（参考） | {RETRAIN['books']['verbmem_f']:.2f} | {RETRAIN['books']['knowmem_f']:.2f} | {RETRAIN['books']['privleak']:.2f} | {RETRAIN['books']['knowmem_r']:.2f} |
| **20260811 本地训练+评** | {cell(books_row, 'verbmem_f')} | {cell(books_row, 'knowmem_f')} | {cell(books_row, 'privleak')} | {cell(books_row, 'knowmem_r')} |

## 各目录摘要

| 目录 | 文件 |
| --- | --- |
| `20251017/` | 论文 Table 2 基线（csv/txt/png + simnpo 字段摘要） |
| `20260811/` | 本地 News/Books 训练评估：`out_*_simnpo_gdr.csv` + `*_summary.txt` |

最新写入语料：**{title}**。
"""
    path.write_text(body, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["news", "books"], required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--hparams", default="")
    ap.add_argument("--train-loss", default="")
    ap.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()

    with args.csv.open() as f:
        row = next(csv.DictReader(f))

    write_summary(
        args.corpus,
        row,
        args.summary,
        ckpt=args.ckpt,
        log=args.log,
        hparams=args.hparams,
        train_loss=args.train_loss or None,
    )
    update_duizhao(args.results_root, args.corpus, row)


if __name__ == "__main__":
    main()
