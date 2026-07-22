"""
绘制 K 消融加速比折线图（启用 KV Cache）。

用法:
  python plot_k_speedup.py
  python plot_k_speedup.py --xlsx "D:/Desktop/复现/data/K消融对比（启用KVCache）.xlsx"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_XLSX = Path(r"D:\Desktop\复现\data\K消融对比（启用KVCache）.xlsx")


def plot_k_speedup(xlsx: Path, out: Path | None = None) -> Path:
    df = pd.read_excel(xlsx)
    required = {"k", "code_speedup", "story_speedup", "facts_speedup"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}，实际列: {list(df.columns)}")

    df = df.sort_values("k")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    ax.plot(df["k"], df["code_speedup"], marker="o", linewidth=2, label="Code (fibonacci)")
    ax.plot(df["k"], df["story_speedup"], marker="s", linewidth=2, label="Story")
    ax.plot(df["k"], df["facts_speedup"], marker="^", linewidth=2, label="Facts")

    # y=1 分界线：>1 加速，<1 降速
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="speedup = 1")

    ax.set_xlabel("Draft length K")
    ax.set_ylabel("Speedup (t_AR / t_SD)")
    ax.set_title("Speculative Decoding: Speedup vs K (with KV Cache)")
    ax.set_xticks(list(df["k"]))
    ax.set_ylim(0, max(2.5, float(df[["code_speedup", "story_speedup", "facts_speedup"]].max().max()) + 0.2))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if out is None:
        out = xlsx.with_name(xlsx.stem + "_speedup.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot K-ablation speedup curves")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Excel 数据路径")
    parser.add_argument("--out", type=Path, default=None, help="输出图片路径（默认与 xlsx 同目录）")
    args = parser.parse_args()

    if not args.xlsx.exists():
        raise FileNotFoundError(f"找不到文件: {args.xlsx}")

    out = plot_k_speedup(args.xlsx, args.out)
    print(f"saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
