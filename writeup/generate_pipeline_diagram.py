#!/usr/bin/env python3
"""
Generate paper-grade pipeline block diagram for AIMO3 writeup using Graphviz.
Output: charts/pipeline_diagram.pdf  +  charts/pipeline_diagram.png

10 stages (0–9):
  0 — Baseline Scoring
  1 — Difficulty Clustering
  2 — SFT Warm-start
  3 — Oracle Trace Collection
  4 — Dr.GRPO Reinforcement Learning
  5 — Evaluation
  6 — Community Dataset Aggregation
  7 — Preference Pair Construction
  8 — ORPO Preference Fine-tuning
  9 — MXFP4 Baking + Kaggle Inference
"""

import graphviz
from pathlib import Path

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

# ── Colour palette (print-safe, colourblind-friendly) ──────────────────────
C_DATA   = "#D6E4F0"   # light blue  — data artefacts
C_PROC   = "#EAF4EA"   # light green — processes / training
C_EVAL   = "#FEF9E7"   # light yellow — evaluation
C_FINAL  = "#FDEDEC"   # light red   — final model / inference
C_BORDER = "#2C3E50"   # dark navy   — all borders
FONT     = "Helvetica"


def make_node(g, name, label, shape="box", style="filled", fillcolor=C_PROC,
              fontsize="11", width="2.2", height="0.55", bold=False):
    fw = "bold" if bold else "normal"
    g.node(name, label=label, shape=shape, style=style, fillcolor=fillcolor,
           color=C_BORDER, fontname=FONT, fontsize=fontsize,
           fontweight=fw, width=width, height=height,
           margin="0.12,0.06")


def make_data(g, name, label, fontsize="10"):
    """Parallelogram = data artefact."""
    g.node(name, label=label, shape="parallelogram", style="filled",
           fillcolor=C_DATA, color=C_BORDER, fontname=FONT,
           fontsize=fontsize, width="2.4", height="0.45", margin="0.12,0.05")


def edge(g, a, b, label="", color="#555555", style="solid", constraint="true"):
    g.edge(a, b, label=label, fontname=FONT, fontsize="9",
           color=color, style=style, arrowsize="0.7", constraint=constraint)


# ── Build diagram ──────────────────────────────────────────────────────────
g = graphviz.Digraph(
    "AIMO3_Pipeline",
    format="pdf",
    graph_attr={
        "rankdir": "TB",
        "splines": "polyline",
        "nodesep": "0.5",
        "ranksep": "0.65",
        "bgcolor": "white",
        "pad": "0.5",
        "fontname": FONT,
        "label": "AIMO3 Training & Inference Pipeline  (Mohammad Shadab Alam)",
        "labelloc": "t",
        "fontsize": "15",
        "fontweight": "bold",
        "newrank": "true",
    },
    node_attr={"fontname": FONT},
    edge_attr={"fontname": FONT},
)

# ── Stage 0: Baseline Scoring ───────────────────────────────────────────────
with g.subgraph(name="cluster_0") as s:
    s.attr(label="Stage 0 — Baseline Scoring", style="rounded,dashed",
           color="#999999", fontsize="10", fontname=FONT)
    make_data(s, "openmath", "OpenMathReasoning\n249,341 integer-answer examples")
    make_node(s, "baseline", "Baseline Evaluation\ngpt-oss-120b  ·  T=0  ·  Harmony API\nGreedy decoding  ·  seed=42",
              fillcolor=C_PROC)
    make_data(s, "pred_log", "predictions_log_base.jsonl\n662 problems  ·  53.0% correct")

edge(g, "openmath", "baseline")
edge(g, "baseline", "pred_log")

# ── Stage 1: Difficulty Clustering ─────────────────────────────────────────
with g.subgraph(name="cluster_1") as s:
    s.attr(label="Stage 1 — Difficulty Clustering",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "cluster",
              "K-Means Clustering\nk=15  ·  all-MiniLM-L6-v2\nthreshold ≥ 0.59 mismatch rate",
              fillcolor=C_PROC)
    make_data(s, "hmclust",
              "high_mismatch_clusters.jsonl\nCluster 3  ·  68.12% mismatch  ·  2,710 problems")

edge(g, "pred_log", "cluster")
edge(g, "cluster",  "hmclust")

# ── Stage 2: SFT Warm-start ────────────────────────────────────────────────
with g.subgraph(name="cluster_2") as s:
    s.attr(label="Stage 2 — SFT Warm-start",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "harmony",
              "Convert to Harmony Format\ntool calls + tool outputs + \\boxed{}",
              fillcolor=C_PROC)
    make_data(s, "hmharm",
              "high_mismatch_harmony.jsonl\n2,710 Harmony-formatted examples")
    make_node(s, "sft",
              "SFT LoRA Warm-start\nLoRA r=4  α=8  ·  gpt-oss-120b BnB4\nExposes model to its failure problems",
              fillcolor=C_PROC, bold=True, height="0.75")

edge(g, "hmclust", "harmony")
edge(g, "harmony", "hmharm")
edge(g, "hmharm",  "sft")

# ── Stage 3: Oracle Trace Collection ───────────────────────────────────────
with g.subgraph(name="cluster_3") as s:
    s.attr(label="Stage 3 — Oracle Trace Collection",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "oracle",
              "Oracle Trace Collection\nvLLM  ·  ≤8 attempts / problem\nretain correct traces only",
              fillcolor=C_PROC)
    make_data(s, "certs",
              "aimo_certs_{151→173→207→219→243→283}.jsonl\n10,002 total  ·  207 used for GRPO")

edge(g, "pred_log", "oracle", style="dashed",
     label=" no-match\nproblems", constraint="false")
edge(g, "cluster",  "oracle", style="dashed",
     label=" kmeans_model.pkl\n(cluster tags)", constraint="false")
edge(g, "oracle",   "certs")
# keep Stage 3 below Stage 2
g.edge("sft", "oracle", style="invis", constraint="true")

# ── Stage 4: Dr.GRPO ────────────────────────────────────────────────────────
with g.subgraph(name="cluster_4") as s:
    s.attr(label="Stage 4 — Dr.GRPO Reinforcement Learning",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "grpo",
              "Dr.GRPO  ·  Two-Phase RL\n"
              "Phase A (10 steps): certificate-first reward\n"
              "Phase B (20 steps): answer-first reward\n"
              "LR=5×10⁻⁶  β=0.001  G=2  batch=1\n"
              "~30 oracle traces / iteration  ·  ≤30 min/iter on H100",
              fillcolor=C_PROC, bold=True, height="1.0")
    make_data(s, "grpo_ada", "GRPO LoRA Adapter\ngpt-oss-120b  +  r=4 adapter")

edge(g, "sft",   "grpo", label=" warm-start\nadapter")
edge(g, "certs", "grpo", label=" 207 oracle\ntraces")
edge(g, "grpo",  "grpo_ada")

# ── Stage 5: Evaluation ─────────────────────────────────────────────────────
with g.subgraph(name="cluster_5") as s:
    s.attr(label="Stage 5 — Evaluation", style="rounded,dashed",
           color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "eval",
              "Hard-50 + held-out evaluation\nbase  →  +SFT  →  +GRPO\nexact-match accuracy",
              fillcolor=C_EVAL)
    make_data(s, "eval_res", "53.0%  →  ?  →  77.3%  (+24.3 pp)")

edge(g, "grpo_ada", "eval")
edge(g, "eval",     "eval_res")

# ── Stage 6: Community Dataset Aggregation ─────────────────────────────────
with g.subgraph(name="cluster_6") as s:
    s.attr(label="Stage 6 — Community Dataset Aggregation",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_data(s, "comm",
              "11 Community Sources\n2,664 problems\n"
              "(CrystalMath 634 · ODA 659 · DAPO 561 · …)")
    make_node(s, "scoring",
              "vLLM Scoring\ngpt-oss-120b  ·  MAX_ATTEMPTS=3\nrecord correct + wrong attempts",
              fillcolor=C_PROC)
    make_data(s, "scored",
              "scored_combined_math.jsonl\n2,664 problems × ≤3 attempts")

edge(g, "comm",    "scoring")
edge(g, "scoring", "scored")
# keep Stage 6 below Stage 5
g.edge("eval_res", "comm", style="invis", constraint="true")

# ── Stage 7: Preference Pair Construction ──────────────────────────────────
with g.subgraph(name="cluster_7") as s:
    s.attr(label="Stage 7 — Preference Pair Construction",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "prepro",
              "Preference Pair Construction\nprompt = problem\n"
              "chosen  = \\boxed{correct answer}\n"
              "rejected = \\boxed{wrong answer}",
              fillcolor=C_PROC, height="0.85")
    make_data(s, "dpo_dat",
              "training_samples_v25232026.jsonl\n1,415 answer-only preference pairs")

edge(g, "scored",  "prepro")
edge(g, "prepro",  "dpo_dat")

# ── Stage 8: ORPO Fine-tuning ───────────────────────────────────────────────
with g.subgraph(name="cluster_8") as s:
    s.attr(label="Stage 8 — ORPO Preference Fine-tuning",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "orpo",
              "ORPO Fine-tuning\n1,000 steps  ·  LR=5×10⁻⁶  β=0.001\n"
              "LoRA r=4  α=8  ·  paged AdamW-8bit  ·  batch=1\n"
              "i500 → +200 steps → i700 (best checkpoint)",
              fillcolor=C_PROC, bold=True, height="0.9")
    make_data(s, "orpo_ada", "ORPO LoRA Adapter (i700)\ngpt-oss-120b  +  GRPO  +  ORPO")

edge(g, "grpo_ada", "orpo", label=" GRPO adapter\n(initialise)")
edge(g, "dpo_dat",  "orpo", label=" 1,415 pairs")
edge(g, "orpo",     "orpo_ada")

# ── Stage 9: MXFP4 Baking ──────────────────────────────────────────────────
with g.subgraph(name="cluster_9a") as s:
    s.attr(label="Stage 9 — MXFP4 Baking",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "bake",
              "LoRA → MXFP4 Merge\nUnsloth save_pretrained_merged\nsave_method='mxfp4'  ·  8× H100 80 GB",
              fillcolor=C_FINAL)
    make_data(s, "baked",
              "gpt-oss-120b-merged-mxfp4-i700\n~61 GB merged checkpoint")

edge(g, "orpo_ada", "bake")
edge(g, "bake",     "baked")

# ── Stage 10: Kaggle Inference + Submission ─────────────────────────────────
with g.subgraph(name="cluster_9b") as s:
    s.attr(label="Stage 10 — Kaggle Inference & Submission",
           style="rounded,dashed", color="#999999", fontsize="10", fontname=FONT)
    make_node(s, "infer",
              "Kaggle Inference\nvLLM  ·  Harmony encoding\ntool-augmented self-consistency\n1× H100 80 GB  ·  tensor-parallel-size=1",
              fillcolor=C_FINAL, bold=True, height="0.9")
    make_data(s, "result",
              "Best Public LB: 42\n85 submissions  ·  mean 38.57  σ 1.67\nRank 1,799 / 4,138")

edge(g, "baked", "infer")
edge(g, "infer", "result")


# ── Render ─────────────────────────────────────────────────────────────────
pdf_path = str(OUT / "pipeline_diagram")
g.render(pdf_path, cleanup=True)
print(f"✓ PDF:  {pdf_path}.pdf")

g.format = "png"
g.graph_attr["dpi"] = "200"
png_path = str(OUT / "pipeline_diagram")
g.render(png_path, cleanup=True)
print(f"✓ PNG:  {png_path}.png")
