"""Render the two derived figures from the archived run JSON.

    python make_figs.py

Reads evidence/main/results.json, writes figs/curve.pdf, figs/curve.png,
figs/cost.pdf and figs/cost.png.  Every number in both panels comes out of
that JSON; the single hand-typed constant is HASH_US below, which the note
quotes without having shipped a timer for it.

figs/bytes.png is NOT produced here -- it was rendered from a model
checkpoint in a separate step and is shipped pre-rendered so that this
directory stays free of binary weights.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
EVIDENCE = os.path.join(HERE, "evidence", "main", "results.json")

# Quoted from the original note: a single dict lookup, 0.2 microseconds.
HASH_US = 0.2
DX = "#2f4b7c"
TX = "#b03a2e"
GREY = "#9a9a9a"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def load():
    with open(EVIDENCE, encoding="utf-8") as fh:
        return json.load(fh)


def fig_curve(run):
    hist = run["history"]
    ep = [r["epoch"] for r in hist]
    tr = [100 * r["train_byte_acc"] for r in hist]
    te = [100 * r["test_byte_acc"] for r in hist]
    trm = [100 * r["train_file_match"] for r in hist]
    tem = [100 * r["test_file_match"] for r in hist]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))

    ax = axes[0]
    ax.plot(ep, tr, "-o", color=DX, ms=3.2, lw=1.2, label="train")
    ax.plot(ep, te, "-s", color=TX, ms=3.0, lw=1.2, label="test")
    ax.annotate("%.2f%%" % te[-1], (ep[-1], te[-1]),
                xytext=(-16, 10), textcoords="offset points",
                fontsize=7, color=TX)
    ax.annotate("100%", (ep[-1], tr[-1]), xytext=(-14, -12),
                textcoords="offset points", fontsize=7, color=DX)
    ax.set_xlabel("epoch")
    ax.set_ylabel("byte accuracy (%)")
    ax.set_ylim(-4, 108)
    ax.set_title("(a) byte accuracy", loc="left")
    ax.legend(loc="center left", fontsize=7)
    ax.grid(axis="y", color="#ececec", lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.step(ep, trm, where="post", color=DX, lw=1.2, label="train")
    ax.step(ep, tem, where="post", color=TX, lw=1.2, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("exact matches (%)")
    ax.set_ylim(-6, 112)
    ax.set_title("(b) whole-sample exact match", loc="left")
    ax.legend(loc="center left", fontsize=7)
    ax.grid(axis="y", color="#ececec", lw=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curve.pdf"))
    fig.savefig(os.path.join(OUT, "curve.png"), dpi=200)
    plt.close(fig)


def fig_cost(run):
    ms_model1 = 1e3 * run["time_model"]["per_sample_bs1"]
    ms_model16 = 1e3 * run["time_model"]["per_sample_bs16_bucketed"]
    ms_lzma = 1e3 * run["time_lzma"]["per_sample"]
    ms_hash = 1e-3 * HASH_US

    rows = [
        ("python lzma.decompress", ms_lzma, 100.0, DX),
        ("LZMA-Hash (64-entry dict)", ms_hash, 0.0, GREY),
        ("this model, bs=16, bucketed", ms_model16, 0.0, "#8fb3d9"),
        ("this model, bs=1", ms_model1, 0.0, "#6f8fbf"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    acc = [r[2] for r in rows]
    cols = [r[3] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.05),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    y = list(range(len(rows)))
    ax.barh(y, vals, color=cols, height=0.6, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("milliseconds per sample (log scale)")
    ax.set_title("(a) inference cost", loc="left")
    ax.set_xlim(1e-5, 1e2)
    for yi, v in zip(y, vals):
        ax.text(v * 1.6, yi, "%.4g ms" % v, va="center", fontsize=6.5)
    ax.grid(axis="x", color="#ececec", lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.barh(y, acc, color=cols, height=0.6, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 112)
    ax.set_xlabel("exact match on test (%)")
    ax.set_title("(b) what it buys", loc="left")
    for yi, v in zip(y, acc):
        ax.text(v + 2.5, yi, "%.0f%%" % v, va="center", fontsize=6.5)
    ax.grid(axis="x", color="#ececec", lw=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cost.pdf"))
    fig.savefig(os.path.join(OUT, "cost.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    run = load()
    fig_curve(run)
    fig_cost(run)
    print("wrote figs/{curve,cost}.{pdf,png}; figs/bytes.png is shipped, not rendered")


if __name__ == "__main__":
    main()
