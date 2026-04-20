#!/usr/bin/env python3
"""
Generate all charts for AIMO3 writeup from confirmed on-disk data.
Run: python generate_charts.py
Output: charts/ directory
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#6B7280"
PURPLE = "#7C3AED"


# ─────────────────────────────────────────────
# Chart 4: Community dataset composition
# ─────────────────────────────────────────────
def chart_community_datasets():
    # From scored_combined_math.jsonl analysis
    sources = {
        "ODA-Math-460k": 659,
        "DAPO-17K": 561,
        "olympiads-ref": 258,
        "Nemotron-Math-V2": 228,
        "Untagged": 634,
        "PaCoRe-Train-8k": 133,
        "IMO-AnswerBench": 67,
        "Omni-MATH": 50,
        "OlymMATH": 25,
        "BeyondAIME": 22,
        "MathArena": 18,
        "Putnam-Axiom": 9,
    }
    total = sum(sources.values())

    # Merge small sources into "Other" for cleaner pie
    threshold = 30
    labels, sizes, explode = [], [], []
    other = 0
    palette = [BLUE, "#3B82F6", "#60A5FA", "#93C5FD", ORANGE, GREEN,
               "#16A34A", PURPLE, "#A78BFA", GRAY, RED, "#F87171"]

    for i, (k, v) in enumerate(sorted(sources.items(), key=lambda x: -x[1])):
        if v >= threshold:
            labels.append(f"{k}\n({v})")
            sizes.append(v)
        else:
            other += v

    if other:
        labels.append(f"Other\n({other})")
        sizes.append(other)

    colors = palette[:len(labels)]

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(linewidth=1.2, edgecolor="white")
    )
    for t in texts: t.set_fontsize(9)
    for at in autotexts: at.set_fontsize(8)

    ax.set_title(f"Community Dataset Composition for ORPO Training\n(Total: {total:,} problems from 11 sources)",
                 fontsize=12, pad=15)
    fig.tight_layout()
    fig.savefig(OUT / "chart4_community_datasets.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 4: community datasets")


# ─────────────────────────────────────────────
# Chart 5: Cluster mismatch rate from predictions_log_base
# ─────────────────────────────────────────────
def chart_cluster_mismatch():
    log_path = Path("/home/malam/wsl-tunix/imo/openmath_data/predictions_log_base.jsonl")
    if not log_path.exists():
        print("  ⚠ predictions_log_base.jsonl not found, skipping chart 5")
        return

    cluster_data = {}  # cluster_id -> {total, mismatch}
    with open(log_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                cid = d.get("predicted_cluster")
                if cid is None:
                    continue
                if cid not in cluster_data:
                    cluster_data[cid] = {"total": 0, "mismatch": 0}
                cluster_data[cid]["total"] += 1
                if d.get("score_match") != "match":
                    cluster_data[cid]["mismatch"] += 1
            except Exception:
                pass

    if not cluster_data:
        print("  ⚠ No cluster data in predictions_log_base, skipping chart 5")
        return

    cluster_ids = sorted(cluster_data.keys())
    mismatch_rates = [cluster_data[c]["mismatch"] / cluster_data[c]["total"] * 100
                      for c in cluster_ids]
    totals = [cluster_data[c]["total"] for c in cluster_ids]

    threshold = 59.0
    colors = [RED if r >= threshold else BLUE for r in mismatch_rates]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f"C{c}" for c in cluster_ids], mismatch_rates,
                  color=colors, edgecolor="white")

    ax.axhline(threshold, color=RED, linewidth=1.5, linestyle="--",
               label=f"High-mismatch threshold ({threshold:.0f}%)")
    ax.set_xlabel("Cluster ID (k=15, all-MiniLM-L6-v2 embeddings)", fontsize=11)
    ax.set_ylabel("Mismatch Rate (%)", fontsize=11)
    ax.set_title("Per-Cluster Mismatch Rate on Base gpt-oss-120b\n(High-mismatch clusters selected for fine-tuning data)",
                 fontsize=11, pad=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)

    # Annotate sample sizes
    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"n={total}", ha="center", va="bottom", fontsize=7, color=GRAY)

    blue_patch = mpatches.Patch(color=BLUE, label="Low-mismatch cluster (kept as-is)")
    red_patch  = mpatches.Patch(color=RED,  label="High-mismatch cluster (selected for training)")
    ax.legend(handles=[blue_patch, red_patch,
                        mpatches.Patch(color="none", label=f"Threshold: {threshold:.0f}%")],
              fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "chart5_cluster_mismatch.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 5: cluster mismatch rates")


# ─────────────────────────────────────────────
# Chart 6: DRGRPO two-phase reward schematic
# ─────────────────────────────────────────────
def chart_drgrpo_phases():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # Phase A reward components
    compA = ["Box\nformat", "Protocol\nheaders", "Exact\nanswer", "Tool call\nconsistency", "Length\npenalty"]
    wA    = [0.5, 0.3, 5.0, 0.7, -0.05]
    colA  = [GREEN if w > 0 else RED for w in wA]

    axes[0].barh(compA, wA, color=colA, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Phase A: Certificate-First Reward\n(10 steps)", fontsize=11)
    axes[0].set_xlabel("Reward Weight", fontsize=10)

    # Phase B reward components
    compB = ["Box\nformat", "Exact\nanswer", "Tool\nok bonus", "Tool\nconsistency", "Length\npenalty"]
    wB    = [0.5, 5.0, 0.7, 0.7, -0.05]
    colB  = [GREEN if w > 0 else RED for w in wB]

    axes[1].barh(compB, wB, color=colB, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Phase B: Answer-First Reward\n(20 steps)", fontsize=11)
    axes[1].set_xlabel("Reward Weight", fontsize=10)

    fig.suptitle("DRGRPO Two-Phase Reward Structure (LR=5e-6, β=0.001, G=2)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "chart6_drgrpo_phases.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 6: DRGRPO phases")


# ─────────────────────────────────────────────
# Chart 7: Pipeline overview (text diagram)
# ─────────────────────────────────────────────
def chart_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        (0.5,  2.5, "OpenMath\nReasoning\n249,341\nexamples",  BLUE),
        (2.5,  2.5, "Base Model\nScoring\n662 problems\n53% acc",  GRAY),
        (4.5,  2.5, "K-Means\nClustering\nk=15\n2,710 selected",  ORANGE),
        (6.5,  2.5, "DRGRPO\nFine-tuning\n30 steps\n77.3% acc",  PURPLE),
        (8.5,  2.5, "ORPO\nFine-tuning\n1,000 steps\n1,415 pairs",  GREEN),
        (10.5, 2.5, "Kaggle\nSubmission\nH100 vLLM\n+ Tools",  RED),
    ]

    for x, y, label, color in stages:
        fancy = mpatches.FancyBboxPatch((x - 0.85, y - 0.9), 1.7, 1.8,
                                         boxstyle="round,pad=0.1",
                                         linewidth=1.5, edgecolor=color,
                                         facecolor=color + "22")
        ax.add_patch(fancy)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=color)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.85
        x2 = stages[i + 1][0] - 0.85
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.8))

    ax.set_title("AIMO3 Full Training and Inference Pipeline\n(Mohammad Shadab Alam)",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "chart7_pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 7: pipeline overview")


# ─────────────────────────────────────────────
# Chart 8: ORPO checkpoint timeline
# ─────────────────────────────────────────────
def chart_checkpoint_timeline():
    # From ls -lt dpo_adapter — dates and names confirmed
    events = [
        ("Feb 26",  "lora-500\n(initial)"),
        ("Feb 28",  "lora-700"),
        ("Mar 19",  "noskip-ckpt-1000"),
        ("Mar 20",  "gold-skip-ckpt-1000"),
        ("Mar 22",  "v22/v3/v4\nckpt-200/300"),
        ("Mar 25",  "v25-ckpt-500\n(pre-final)"),
        ("Mar 25",  "checkpoint-1000\n(final ORPO)"),
    ]
    x = list(range(len(events)))
    labels = [e[1] for e in events]
    dates  = [e[0] for e in events]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(x, [1]*len(x), color=BLUE, linewidth=2, zorder=1)
    ax.scatter(x, [1]*len(x), s=120, color=BLUE, zorder=2, edgecolors="white", linewidth=1.5)

    for i, (xi, label, date) in enumerate(zip(x, labels, dates)):
        va = "bottom" if i % 2 == 0 else "top"
        offset = 0.12 if va == "bottom" else -0.12
        ax.text(xi, 1 + offset, label, ha="center", va=va, fontsize=8.5,
                fontweight="bold" if "final" in label.lower() else "normal")
        ax.text(xi, 1 - (0.23 if va == "top" else -0.23), date,
                ha="center", va="top" if va == "bottom" else "bottom",
                fontsize=8, color=GRAY)

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0.5, 1.7)
    ax.axis("off")
    ax.set_title("ORPO Fine-tuning Checkpoint Timeline (Feb–Mar 2026)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "chart8_checkpoint_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 8: checkpoint timeline")


# ─────────────────────────────────────────────
# Chart 9: Submission history — phase progression
# ─────────────────────────────────────────────
def chart_submission_scores():
    import statistics

    CSV_PATH = Path(__file__).parent / "aimo3_writeup_supplement.csv"
    if not CSV_PATH.exists():
        print("  ⚠ aimo3_writeup_supplement.csv not found, skipping chart 9")
        return

    BAD_KEYWORDS = ["bad attempt"]
    # dpo_v4 is a sub-category carved out of "dpo" for the final notebook+version
    PHASE_ORDER = [
        "initial exploration",
        "initial exploration - lora",
        "rag",
        "lora + rag",
        "rag+baked lora weights",
        "dpo + rag",
        "dpo_other",
        "dpo_v4",
    ]
    PHASE_LABELS = [
        "Initial\nExploration",
        "Init. Exp.\n+ LoRA",
        "RAG\n(base)",
        "LoRA\n+ RAG",
        "Baked LoRA\n+ RAG",
        "DPO\n+ RAG",
        "DPO\n(other ver.)",
        "DPO v4\n(final)",
    ]
    PHASE_COLORS = [GRAY, ORANGE, BLUE, "#60A5FA", PURPLE, GREEN, "#F59E0B", RED]

    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("Score") or not r["Score"].strip():
                continue
            try:
                score = int(r["Score"])
            except ValueError:
                continue
            phase = (r.get("phase") or "").strip().lower()
            version = (r.get("version") or "").strip().lower()
            name = (r.get("submission_name") or "").strip().lower()
            is_bad = any(k in phase for k in BAD_KEYWORDS)
            # Reclassify DPO rows: v4 of the final notebook gets its own sub-phase
            if phase == "dpo" and version == "v4" and "c283" in name:
                effective_phase = "dpo_v4"
            elif phase == "dpo":
                effective_phase = "dpo_other"
            else:
                effective_phase = phase
            rows.append({"sno": int(r["sno"]) if r.get("sno") else 0,
                         "phase": effective_phase, "score": score, "is_bad": is_bad})

    valid = [r for r in rows if not r["is_bad"]]
    bad   = [r for r in rows if r["is_bad"]]

    # Per-phase stats (valid only)
    phase_stats = {}
    for p in PHASE_ORDER:
        scores = [r["score"] for r in valid if r["phase"] == p]
        if scores:
            phase_stats[p] = {
                "mean": statistics.mean(scores),
                "std":  statistics.stdev(scores) if len(scores) > 1 else 0,
                "max":  max(scores),
                "n":    len(scores),
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # ── Left panel: scatter over submission index ──────────────────
    color_map = dict(zip(PHASE_ORDER, PHASE_COLORS))
    for r in valid:
        c = color_map.get(r["phase"], GRAY)
        ax1.scatter(r["sno"], r["score"], color=c, s=40, alpha=0.8, zorder=3)
    for r in bad:
        ax1.scatter(r["sno"], r["score"], marker="x", color="#9CA3AF", s=50,
                    linewidths=1.5, zorder=2, label="_nolegend_")

    # Phase region backgrounds
    phase_sno = {p: [r["sno"] for r in valid if r["phase"] == p] for p in PHASE_ORDER}
    for p, c in zip(PHASE_ORDER, PHASE_COLORS):
        snos = phase_sno.get(p, [])
        if snos:
            ax1.axvspan(min(snos) - 0.8, max(snos) + 0.8, alpha=0.07, color=c, zorder=1)

    # Mark overall best
    best = max(valid, key=lambda r: r["score"])
    ax1.scatter(best["sno"], best["score"], marker="*", color="gold",
                s=250, zorder=5, edgecolors="black", linewidths=0.8)
    ax1.annotate(f"Best: {best['score']}", xy=(best["sno"], best["score"]),
                 xytext=(best["sno"] + 2, best["score"] + 0.5),
                 fontsize=9, fontweight="bold", color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Legend patches
    patches = [mpatches.Patch(color=c, label=l, alpha=0.8)
               for p, c, l in zip(PHASE_ORDER, PHASE_COLORS, PHASE_LABELS)]
    patches.append(mpatches.Patch(color="#9CA3AF", label="Bad attempt\n(excluded)", alpha=0.5))
    ax1.legend(handles=patches, fontsize=7.5, ncol=2, loc="lower right",
               framealpha=0.9)

    ax1.set_xlabel("Submission Index", fontsize=11)
    ax1.set_ylabel("Public LB Score", fontsize=11)
    ax1.set_ylim(20, 50)
    ax1.set_title("All Submissions by Phase (× = bad attempt excluded from analysis)",
                  fontsize=10.5, pad=8)

    # ── Right panel: per-phase mean ± std ──────────────────────────
    x_pos = range(len(PHASE_ORDER))
    means = [phase_stats.get(p, {}).get("mean", 0) for p in PHASE_ORDER]
    stds  = [phase_stats.get(p, {}).get("std",  0) for p in PHASE_ORDER]
    ns    = [phase_stats.get(p, {}).get("n",    0) for p in PHASE_ORDER]

    bars = ax2.bar(x_pos, means, color=PHASE_COLORS, width=0.6,
                   edgecolor="white", linewidth=1.2, alpha=0.85)
    ax2.errorbar(x_pos, means, yerr=stds, fmt="none",
                 color="black", capsize=4, linewidth=1.5, capthick=1.5)

    for xi, (mean, n) in enumerate(zip(means, ns)):
        if n:
            ax2.text(xi, mean + stds[xi] + 0.3, f"{mean:.1f}\n(n={n})",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(PHASE_LABELS, fontsize=8)
    ax2.set_ylabel("Mean Public LB Score", fontsize=11)
    ax2.set_ylim(28, 46)
    ax2.set_title("Per-Phase Score Statistics\n(mean ± std, bad attempts excluded)",
                  fontsize=10.5, pad=8)
    ax2.axhline(36.6, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("AIMO3 Submission History: Phase-by-Phase Leaderboard Progression",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "chart9_submission_scores.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 9: submission scores by phase")


# ─────────────────────────────────────────────
# Chart 11: Final submission (v4) score distribution — horizontal
# ─────────────────────────────────────────────
def chart_v4_score_distribution():
    import statistics

    CSV_PATH = Path(__file__).parent / "aimo3_writeup_supplement.csv"
    if not CSV_PATH.exists():
        print("  ⚠ CSV not found, skipping chart 11")
        return

    scores = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("Score") or not r["Score"].strip():
                continue
            phase = (r.get("phase") or "").strip().lower()
            version = (r.get("version") or "").strip().lower()
            name = (r.get("submission_name") or "").strip().lower()
            if any(k in phase for k in ["bad attempt"]):
                continue
            if phase == "dpo" and version == "v4" and "c283" in name:
                try:
                    scores.append(int(r["Score"]))
                except ValueError:
                    pass

    if not scores:
        print("  ⚠ No v4 scores found, skipping chart 11")
        return

    mean  = statistics.mean(scores)
    std   = statistics.stdev(scores)
    max_s = max(scores)

    fig, ax = plt.subplots(figsize=(11, 3.2))

    # Shaded ±2σ band
    ax.axhspan(mean - 2*std, mean + 2*std, color=RED, alpha=0.08,
               label=f"Mean ± 2σ  ({mean-2*std:.1f} – {mean+2*std:.1f})")

    # ±1σ band
    ax.axhspan(mean - std, mean + std, color=RED, alpha=0.12,
               label=f"Mean ± 1σ  ({mean-std:.1f} – {mean+std:.1f})")

    # Mean line
    ax.axhline(mean, color=RED, linewidth=2, linestyle="-",
               label=f"Mean = {mean:.2f}")

    # Individual scores — jittered x so overlapping points separate
    np.random.seed(42)
    jitter = np.random.uniform(-0.18, 0.18, len(scores))
    colors_pts = [RED if s == max_s else BLUE for s in scores]
    sizes      = [120 if s == max_s else 45  for s in scores]
    zorders    = [5   if s == max_s else 3   for s in scores]

    for x, s, c, sz, z in zip(jitter, scores, colors_pts, sizes, zorders):
        ax.scatter(x, s, color=c, s=sz, alpha=0.85, zorder=z,
                   edgecolors="white", linewidth=0.6)

    # Annotate max
    max_x = jitter[scores.index(max_s)]
    ax.annotate(f"Max = {max_s}", xy=(max_x, max_s),
                xytext=(0.28, max_s),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
                fontsize=10, fontweight="bold", color=RED)

    # Annotate mean
    ax.text(0.55, mean + 0.15, f"μ = {mean:.2f}", color=RED,
            fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.55, mean - 2*std - 0.3, f"σ = {std:.2f}", color=GRAY,
            fontsize=9, va="top")

    ax.set_xlim(-0.6, 0.7)
    y_pad = 1.5
    ax.set_ylim(min(scores) - y_pad, max_s + y_pad)
    ax.set_ylabel("Public LB Score", fontsize=12)
    ax.set_xticks([])
    ax.set_title(
        f"Final Submission Score Distribution — aimo_rag_baked_c283 v4\n"
        f"(n={len(scores)}, model: gpt-oss-120b-merged-mxfp4-i700, no RAG)",
        fontsize=11, pad=10
    )

    handles = [
        mpatches.Patch(color=RED, alpha=0.20, label=f"±2σ band  [{mean-2*std:.1f}, {mean+2*std:.1f}]"),
        mpatches.Patch(color=RED, alpha=0.32, label=f"±1σ band  [{mean-std:.1f}, {mean+std:.1f}]"),
        plt.Line2D([0], [0], color=RED, lw=2, label=f"Mean = {mean:.2f}"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
                   markersize=9, label=f"Max = {max_s} (highlighted)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                   markersize=7, label="Individual submissions"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "chart11_v4_score_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 11: v4 score distribution")


def chart_hard50_ablation():
    """
    Two-panel chart: hard-50 scores + runtimes per model checkpoint.
    Data from writeup/hard-50.csv.

    Elimination story (two criteria, no format distinction):
      1. i500, i1k-gold eliminated — public LB score lower than i700
      2. v23, v25 eliminated — slower runtimes
      3. i700 selected — best public LB + acceptable runtime
    """
    import statistics

    DATA = [
        ("i500",       "v3",  201, 34),
        ("i500",       "v4",  188, 29),
        ("i700",       "v6",  173, 26),
        ("i700",       "v7",  177, 28),
        ("i1k-gold",   "v8",  169, 28),
        ("i1k-gold",   "v9",  176, 29),
        ("v23032026",  "v10", 190, 27),
        ("v23032026",  "v11", 195, 26),
        ("v23032026",  "v12", 175, 23),
        ("v25032026",  "v15", 187, 26),
        ("v25032026",  "v16", 177, 30),
    ]

    # Order: i500 first (LB eliminated), then remaining by descending runtime (time eliminated)
    MODEL_ORDER = ["i500", "v23032026", "v25032026", "i1k-gold", "i700"]
    SHORT = {
        "i500":      "i500",
        "i700":      "i700\n★ selected",
        "i1k-gold":  "i1k-gold",
        "v23032026": "v23032026",
        "v25032026": "v25032026",
    }
    # red = eliminated by LB, orange = eliminated by runtime, green = selected
    COLORS = {
        "i500":      RED,
        "i700":      GREEN,
        "i1k-gold":  RED,
        "v23032026": ORANGE,
        "v25032026": ORANGE,
    }

    from collections import defaultdict
    grouped = defaultdict(lambda: {"scores": [], "times": []})
    for model, _run, t, s in DATA:
        grouped[model]["scores"].append(s)
        grouped[model]["times"].append(t)

    means_s = {m: statistics.mean(grouped[m]["scores"]) for m in MODEL_ORDER}
    means_t = {m: statistics.mean(grouped[m]["times"])  for m in MODEL_ORDER}

    x = list(range(len(MODEL_ORDER)))
    xlabels = [SHORT[m] for m in MODEL_ORDER]
    bar_colors = [COLORS[m] for m in MODEL_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: scores ────────────────────────────────────────────────
    ax1.bar(x, [means_s[m] for m in MODEL_ORDER],
            color=bar_colors, width=0.55, alpha=0.75,
            edgecolor="white", linewidth=1.2)

    np.random.seed(0)
    for i, m in enumerate(MODEL_ORDER):
        jitter = np.random.uniform(-0.14, 0.14, len(grouped[m]["scores"]))
        ax1.scatter(i + jitter, grouped[m]["scores"],
                    color=COLORS[m], s=55, zorder=4,
                    edgecolors="white", linewidth=0.6, alpha=0.9)

    for xi, m in zip(x, MODEL_ORDER):
        ax1.text(xi, means_s[m] + 0.5, f"{means_s[m]:.1f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=COLORS[m])

    ax1.annotate("① Eliminated:\npublic LB score ↓",
                 xy=(0, means_s["i500"]),
                 xytext=(0.6, means_s["i500"] - 5),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax1.annotate("① Eliminated:\npublic LB score ↓",
                 xy=(3, means_s["i1k-gold"]),
                 xytext=(1.8, means_s["i1k-gold"] + 4),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax1.annotate("② Eliminated:\nruntime ↑",
                 xy=(1, means_s["v23032026"]),
                 xytext=(0.0, means_s["v23032026"] + 5),
                 fontsize=8, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=9)
    ax1.set_ylabel("Hard-50 Score  (out of 50)", fontsize=11)
    ax1.set_ylim(0, 42)
    ax1.set_title("Hard-50 Benchmark Scores by Checkpoint\n(dots = individual runs, bar = mean)",
                  fontsize=10.5, pad=8)

    # ── Right: runtimes ─────────────────────────────────────────────
    ax2.bar(x, [means_t[m] for m in MODEL_ORDER],
            color=bar_colors, width=0.55, alpha=0.75,
            edgecolor="white", linewidth=1.2)

    for i, m in enumerate(MODEL_ORDER):
        jitter = np.random.uniform(-0.14, 0.14, len(grouped[m]["times"]))
        ax2.scatter(i + jitter, grouped[m]["times"],
                    color=COLORS[m], s=55, zorder=4,
                    edgecolors="white", linewidth=0.6, alpha=0.9)

    for xi, m in zip(x, MODEL_ORDER):
        ax2.text(xi, means_t[m] + 1, f"{means_t[m]:.0f}m",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=COLORS[m])

    # Mark i700 as fastest acceptable
    ax2.annotate("Fastest\nacceptable",
                 xy=(4, means_t["i700"]),
                 xytext=(3.1, means_t["i700"] - 10),
                 fontsize=8, color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=9)
    ax2.set_ylabel("Hard-50 Runtime  (minutes)", fontsize=11)
    ax2.set_ylim(140, 220)
    ax2.set_title("Hard-50 Runtime by Checkpoint\n(i500 & i1k-gold excluded by LB; i700 best among remainder)",
                  fontsize=10.5, pad=8)

    legend_handles = [
        mpatches.Patch(color=RED,    alpha=0.8, label="① Eliminated — public LB score lower than i700"),
        mpatches.Patch(color=ORANGE, alpha=0.8, label="② Eliminated — slower runtime"),
        mpatches.Patch(color=GREEN,  alpha=0.8, label="Selected — i700"),
    ]
    fig.legend(handles=legend_handles, fontsize=9, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    fig.suptitle(
        "Checkpoint Elimination: Public LB Score (criterion 1) then Runtime (criterion 2)\n"
        "(Hard-50 benchmark: ZFTurbo hard-math-problems-for-aimo-3, 50 problems)",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "chart10_hard50_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Chart 10: hard-50 ablation")


if __name__ == "__main__":
    print("Generating AIMO3 writeup charts...\n")
    chart_community_datasets()
    chart_cluster_mismatch()
    chart_drgrpo_phases()
    chart_pipeline()
    chart_checkpoint_timeline()
    chart_submission_scores()
    chart_v4_score_distribution()
    chart_hard50_ablation()
    print(f"\nAll charts saved to {OUT}/")
