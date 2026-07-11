"""
Comprehensive experiment score visualization generator.
Reads score files from baseline/, chunk_page/, final_rewrite_hyde/ and produces:
  - Summary tables (CSV + console)
  - Bar charts for each dataset type (single_asr, single_text, multi_asr, multi_text, dl)
  - Overall score comparison across experiments
  - Dedicated BERTScore comparison chart
  - Dedicated Latency comparison chart
  - Per-experiment-pair (baseline vs chunk_page vs final) radar/spider charts
"""
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOTS = [Path("baseline"), Path("chunk_page"), Path("final_rewrite_hyde")]
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mapping from internal keys to human-readable labels
CATEGORY_LABELS: Dict[str, str] = {
    "single_asr": "Single-Turn ASR",
    "single_text": "Single-Turn Text",
    "multi_asr": "Multi-Turn ASR",
    "multi_text": "Multi-Turn Text",
    "dl": "DL Exam",
}

# Sections to look for in each score file
SECTION_PATTERNS: Dict[str, str] = {
    "single_asr": "Single-Turn Dialogue [With ASR Audio Stream]",
    "single_text": "Single-Turn Dialogue [Pure Text Standard Stream]",
    "multi_asr": "Multi-Turn Dialogue [With ASR Audio Stream]",
    "multi_text": "Multi-Turn Dialogue [Pure Text Standard Stream]",
    "dl": "Deep Learning Exam [Pure Text No-RAG Stream]",
}

# Metric keys we care about
METRIC_KEYS: List[str] = [
    "bert", "correctness", "groundedness",
    "retrieval_relevance", "answer_relevance",
    "context_precision", "context_recall",
    "latency", "wer",
]

# Display names for metrics
METRIC_DISPLAY: Dict[str, str] = {
    "bert": "BERTScore F1",
    "correctness": "Answer Correctness",
    "groundedness": "Groundedness",
    "retrieval_relevance": "Retrieval Relevance",
    "answer_relevance": "Answer Relevance",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "latency": "Latency (s)",
    "wer": "WER",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def find_score_files() -> List[Path]:
    files = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() in {"scores", "score"}:
                files.append(path)
    return sorted(files)


def extract_name(text: str) -> str:
    """Extract the 'name : xxx' line if present."""
    m = re.search(r"name\s*:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return ""


def parse_metric_value(section_text: str, metric_name: str) -> Optional[float]:
    """Parse a single metric from a section of text."""
    if metric_name == "latency":
        m = re.search(r"Execution Time Cost \(s\)\s*:\s*([0-9.]+)s", section_text)
        return float(m.group(1)) if m else None
    if metric_name == "wer":
        m = re.search(r"Word Error Rate \(WER\)\s*:\s*([0-9.]+)", section_text)
        return float(m.group(1)) if m else None
    patterns = {
        "bert": r"BERTScore F1 Similarity\s*:\s*([0-9.]+)",
        "correctness": r"Answer Correctness\s*:\s*([0-9.]+)",
        "groundedness": r"Groundedness \(Anti-Hal\)\s*:\s*([0-9.]+)",
        "retrieval_relevance": r"Retrieval Relevance\s*:\s*([0-9.]+)",
        "answer_relevance": r"Answer Relevance\s*:\s*([0-9.]+)",
        "context_precision": r"Context Precision\s*:\s*([0-9.]+)",
        "context_recall": r"Context Recall\s*:\s*([0-9.]+)",
    }
    m = re.search(patterns.get(metric_name, ""), section_text)
    return float(m.group(1)) if m else None


def parse_category(text: str, category: str, marker: str) -> Dict[str, float]:
    """Extract all metrics for a given section marker."""
    start = text.find(marker)
    if start == -1:
        return {}
    end = len(text)
    for other_marker in SECTION_PATTERNS.values():
        if other_marker != marker:
            nxt = text.find(other_marker, start + 1)
            if nxt != -1 and nxt < end:
                end = nxt
    section_text = text[start:end]
    metrics = {}
    for mk in METRIC_KEYS:
        v = parse_metric_value(section_text, mk)
        if v is not None:
            metrics[mk] = v
    return metrics


def parse_score_file(path: Path) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Returns (experiment_name, {category: {metric: value}}).
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    exp_name = extract_name(text)
    if not exp_name:
        # fallback: use parent folder name
        exp_name = path.parent.name
        # add grandparent folder if not already descriptive
        if path.parent.parent.name.lower() in ("baseline", "chunk_page") and len(path.parent.parent.name) > 0:
            exp_name = f"{path.parent.parent.name}/{path.parent.name}"

    data: Dict[str, Dict[str, float]] = {}
    for category, marker in SECTION_PATTERNS.items():
        metrics = parse_category(text, category, marker)
        if metrics:
            data[category] = metrics
    return exp_name, data


def collect_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Returns {experiment_name: {category: {metric: value}}}."""
    results = {}
    for path in find_score_files():
        try:
            exp_name, data = parse_score_file(path)
        except Exception as exc:
            print(f"  [SKIP] {path}: {exc}")
            continue
        if data:
            if exp_name in results:
                # merge (shouldn't happen, but be safe)
                results[exp_name].update(data)
            else:
                results[exp_name] = data
            print(f"  [OK]   {exp_name} ({path}) -> {list(data.keys())}")
        else:
            print(f"  [WARN] {path} -> no parseable sections")
    return results


def get_short_name(exp_name: str) -> str:
    """Shorten experiment names for display."""
    parts = exp_name.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-2].lower() in ("baseline", "chunk_page"):
        return parts[-1]
    return exp_name


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _bar_chart(ax, labels, values, title, ylabel, color="#4C78A8", rotate=30):
    """Draw a bar chart on a given axis."""
    short = [get_short_name(l) for l in labels]
    bars = ax.bar(range(len(labels)), values, color=color, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=rotate, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    # Add value labels on bars (if not too many)
    if len(values) <= 20:
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)


def _grouped_bar_chart(ax, experiments, category_keys, metric_key, title, ylabel, colors=None):
    """
    Grouped bar chart: one group per experiment, one bar per category (ASR vs Text etc.)
    """
    short_names = [get_short_name(e) for e in experiments]
    n_exp = len(experiments)
    n_cat = len(category_keys)
    index = np.arange(n_exp)
    bar_width = 0.8 / n_cat

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_cat))

    for i, cat in enumerate(category_keys):
        vals = []
        for exp in experiments:
            v = results.get(exp, {}).get(cat, {}).get(metric_key)
            vals.append(v if v is not None else 0)
        offset = (i - (n_cat - 1) / 2) * bar_width
        bars = ax.bar(index + offset, vals, bar_width, label=CATEGORY_LABELS.get(cat, cat),
                      color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=0)

    ax.set_xticks(index)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


# =============================================
# 1. Overall per-dataset-type metric panels
# =============================================
def plot_category_panels(results: Dict):
    """For each dataset type, show ALL metrics in one figure."""
    for cat_key, cat_label in CATEGORY_LABELS.items():
        # Collect all experiments that have data for this category
        exp_list = sorted([e for e in results if cat_key in results[e]])
        if not exp_list:
            print(f"  No data for {cat_label}, skip")
            continue

        # Determine which metrics to show for this category
        sample = results[exp_list[0]][cat_key]
        available_metrics = [m for m in METRIC_KEYS if m in sample]

        n_metrics = len(available_metrics)
        if n_metrics == 0:
            continue

        cols = min(4, n_metrics)
        rows = math.ceil(n_metrics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        fig.suptitle(f"{cat_label} — All Metrics", fontsize=16, fontweight="bold", y=1.02)

        for i, mk in enumerate(available_metrics):
            ax = axes[i]
            exp_names = []
            values = []
            for exp in exp_list:
                v = results[exp][cat_key].get(mk)
                if v is not None and not math.isnan(v):
                    exp_names.append(exp)
                    values.append(v)
            if exp_names:
                _bar_chart(ax, exp_names, values,
                           METRIC_DISPLAY.get(mk, mk),
                           METRIC_DISPLAY.get(mk, mk))
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(METRIC_DISPLAY.get(mk, mk))

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{cat_key}_all_metrics.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


# =============================================
# 2. Overall score comparison (by experiment)
# =============================================
def compute_overall_score(metrics: Dict[str, float], kind: str) -> float:
    """Normalized average score across metrics (excluding latency/WER)."""
    if kind == "dl":
        keys = ["bert", "correctness", "answer_relevance"]
    else:
        keys = ["bert", "correctness", "groundedness",
                "retrieval_relevance", "answer_relevance",
                "context_precision", "context_recall"]
    vals = [metrics[k] for k in keys if k in metrics and not math.isnan(metrics[k])]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def plot_overall_score_comparison(results: Dict):
    """Bar chart: overall composite score per experiment, grouped by dataset type."""
    categories_to_plot = [
        ("single_asr", "Single-Turn ASR"),
        ("single_text", "Single-Turn Text"),
        ("multi_asr", "Multi-Turn ASR"),
        ("multi_text", "Multi-Turn Text"),
        ("dl", "DL Exam"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # Group 1: single-turn, Group 2: multi-turn, Group 3: DL
    groups = [
        (["single_asr", "single_text"], "Single-Turn Overall"),
        (["multi_asr", "multi_text"], "Multi-Turn Overall"),
        (["dl"], "DL Exam Overall"),
    ]
    colors = ["#2CA02C", "#FF7F0E", "#1F77B4"]

    for ax, (group_cats, group_title) in zip(axes, groups):
        labels = []
        values = []
        for exp in sorted(results):
            for cat_key in group_cats:
                metrics = results[exp].get(cat_key)
                if not metrics:
                    continue
                score = compute_overall_score(metrics, cat_key if cat_key == "dl" else "other")
                if score > 0:
                    labels.append(f"{get_short_name(exp)} | {CATEGORY_LABELS[cat_key]}")
                    values.append(score)
        if labels:
            _bar_chart(ax, labels, values, group_title, "Composite Score", color=colors[groups.index((group_cats, group_title))])
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(group_title)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "overall_score_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# =============================================
# 3. BERTScore comparison (dedicated)
# =============================================
def plot_bertscore_comparison(results: Dict):
    """Dedicated chart: BERTScore for each category across all experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    groups = [
        (["single_asr", "single_text"], "BERTScore — Single Turn"),
        (["multi_asr", "multi_text"], "BERTScore — Multi Turn"),
        (["dl"], "BERTScore — DL Exam"),
    ]
    colors = ["#D62728", "#9467BD", "#8C564B"]

    for ax, (group_cats, group_title) in zip(axes, groups):
        labels = []
        values = []
        for exp in sorted(results):
            for cat_key in group_cats:
                metrics = results[exp].get(cat_key)
                if metrics and "bert" in metrics:
                    labels.append(f"{get_short_name(exp)} | {CATEGORY_LABELS[cat_key]}")
                    values.append(metrics["bert"])
        if labels:
            _bar_chart(ax, labels, values, group_title, "BERTScore F1", color=colors[groups.index((group_cats, group_title))])
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(group_title)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "bertscore_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# =============================================
# 4. Latency comparison (dedicated)
# =============================================
def plot_latency_comparison(results: Dict):
    """Dedicated chart: latency per category across all experiments."""
    all_categories = ["single_asr", "single_text", "multi_asr", "multi_text", "dl"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    groups = [
        (["single_asr", "single_text"], "Latency — Single Turn"),
        (["multi_asr", "multi_text"], "Latency — Multi Turn"),
        (["dl"], "Latency — DL Exam"),
    ]
    colors = ["#F58518", "#17BECF", "#BCBD22"]

    for ax, (group_cats, group_title) in zip(axes, groups):
        labels = []
        values = []
        for exp in sorted(results):
            for cat_key in group_cats:
                metrics = results[exp].get(cat_key)
                if metrics and "latency" in metrics:
                    labels.append(f"{get_short_name(exp)} | {CATEGORY_LABELS[cat_key]}")
                    values.append(metrics["latency"])
        if labels:
            _bar_chart(ax, labels, values, group_title, "Latency (s)", color=colors[groups.index((group_cats, group_title))])
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(group_title)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "latency_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# =============================================
# 5. Per-dataset radar chart
# =============================================
def plot_radar_charts(results: Dict):
    """Radar chart for each dataset type, comparing all experiment approaches."""
    for cat_key, cat_label in CATEGORY_LABELS.items():
        exp_list = sorted([e for e in results if cat_key in results[e]])
        if len(exp_list) < 2:
            continue
        # Metrics to show on radar (exclude latency and WER for radar)
        sample = results[exp_list[0]][cat_key]
        radar_metrics = [m for m in METRIC_KEYS if m in sample and m not in ("latency", "wer")]
        if len(radar_metrics) < 3:
            continue

        n_metrics = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.suptitle(f"Radar Chart — {cat_label}", fontsize=14, fontweight="bold")

        colors = plt.cm.Set2(np.linspace(0, 1, len(exp_list)))

        for idx, exp in enumerate(exp_list):
            metrics = results[exp][cat_key]
            vals = [metrics.get(m, 0) for m in radar_metrics]
            # Normalize scores to 0-1 range where possible
            vals_norm = []
            for m, v in zip(radar_metrics, vals):
                if m == "latency":
                    continue  # skip
                vals_norm.append(v)
            vals_norm += vals_norm[:1]  # close
            ax.plot(angles, vals_norm, "o-", linewidth=2, label=get_short_name(exp), color=colors[idx])
            ax.fill(angles, vals_norm, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_DISPLAY.get(m, m) for m in radar_metrics], fontsize=10)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{cat_key}_radar.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


# =============================================
# 6. Heatmap: experiments x metrics (per dataset type)
# =============================================
def plot_heatmaps(results: Dict):
    """Heatmap for each dataset type: rows=experiments, cols=metrics."""
    for cat_key, cat_label in CATEGORY_LABELS.items():
        exp_list = sorted([e for e in results if cat_key in results[e]])
        if not exp_list:
            continue
        sample = results[exp_list[0]][cat_key]
        metrics_to_show = [m for m in METRIC_KEYS if m in sample and m != "wer"]
        if not metrics_to_show:
            continue

        data = []
        row_labels = []
        for exp in exp_list:
            row = []
            for mk in metrics_to_show:
                v = results[exp][cat_key].get(mk)
                row.append(v if v is not None else 0)
            data.append(row)
            row_labels.append(get_short_name(exp))

        data_arr = np.array(data)
        col_labels = [METRIC_DISPLAY.get(m, m) for m in metrics_to_show]

        fig, ax = plt.subplots(figsize=(max(10, len(metrics_to_show) * 2.5), max(6, len(exp_list) * 0.8)))
        im = ax.imshow(data_arr, cmap="YlGn", aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(f"Score Heatmap — {cat_label}", fontsize=14, fontweight="bold")

        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(j, i, f"{data_arr[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if data_arr[i, j] < 3 else "white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{cat_key}_heatmap.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


# =============================================
# 7. Summary table (CSV)
# =============================================
def write_summary_csv(results: Dict):
    """Write a CSV with all results."""
    rows = []
    header = ["Experiment", "Dataset", "Category"]
    # Collect all metrics found
    all_metrics = set()
    for exp_data in results.values():
        for cat_data in exp_data.values():
            all_metrics.update(cat_data.keys())
    all_metrics = sorted([m for m in all_metrics if m != "wer"])
    header += [METRIC_DISPLAY.get(m, m) for m in all_metrics]

    rows.append(header)

    for exp in sorted(results):
        for cat_key in sorted(results[exp]):
            cat_data = results[exp][cat_key]
            row = [exp, cat_key, CATEGORY_LABELS.get(cat_key, cat_key)]
            for mk in all_metrics:
                v = cat_data.get(mk, "")
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            rows.append(row)

    csv_path = OUTPUT_DIR / "all_scores_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"  Saved {csv_path}")

    # Also print to console
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    for row in rows:
        print(" | ".join(row))


# =============================================
# 8. Separate charts: Single-turn (ASR vs Text on same chart), Multi-turn (ASR vs Text)
# =============================================
def plot_grouped_dataset_comparison(results: Dict):
    """
    Grouped bar charts: for single-turn (ASR + Text side-by-side) and multi-turn (ASR + Text)
    showing BERTScore, Correctness, Groundedness, Retrieval Relevance, Answer Relevance,
    Context Precision, Context Recall, Latency.
    """
    groups = [
        (["single_asr", "single_text"], "Single-Turn ASR vs Text"),
        (["multi_asr", "multi_text"], "Multi-Turn ASR vs Text"),
    ]

    for group_cats, group_title in groups:
        exp_list = sorted([e for e in results if all(c in results[e] for c in group_cats)])
        if not exp_list:
            continue

        sample = results[exp_list[0]][group_cats[0]]
        metrics_to_plot = [m for m in METRIC_KEYS if m in sample and m != "wer"]

        n_metrics = len(metrics_to_plot)
        cols = min(3, n_metrics)
        rows = math.ceil(n_metrics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        fig.suptitle(group_title, fontsize=16, fontweight="bold", y=1.02)

        colors = ["#1F77B4", "#FF7F0E"]  # ASR, Text

        for i, mk in enumerate(metrics_to_plot):
            ax = axes[i]
            _grouped_bar_chart(ax, exp_list, group_cats, mk,
                               METRIC_DISPLAY.get(mk, mk),
                               METRIC_DISPLAY.get(mk, mk),
                               colors=colors)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{group_title.replace(' ', '_').lower()}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


# =============================================
# 9. Separate: DL exam metrics + BERTScore & Latency highlight
# =============================================
def plot_dl_highlight(results: Dict):
    """DL exam: all metrics + dedicated BERTScore + Latency comparison."""
    exp_list = sorted([e for e in results if "dl" in results[e]])
    if not exp_list:
        return

    # --- DL All metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("DL Exam — Detailed Metrics", fontsize=16, fontweight="bold")

    dl_metrics = ["bert", "correctness", "answer_relevance", "latency"]
    colors = ["#D62728", "#2CA02C", "#9467BD", "#F58518"]
    for i, (mk, col) in enumerate(zip(dl_metrics, colors)):
        ax = axes[i]
        labels = []
        values = []
        for exp in exp_list:
            v = results[exp]["dl"].get(mk)
            if v is not None:
                labels.append(get_short_name(exp))
                values.append(v)
        if labels:
            _bar_chart(ax, labels, values, METRIC_DISPLAY.get(mk, mk),
                       METRIC_DISPLAY.get(mk, mk), color=col)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(METRIC_DISPLAY.get(mk, mk))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "dl_exam_detailed.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    # --- DL BERTScore compared with single_text and multi_text ---
    fig, ax = plt.subplots(figsize=(14, 6))
    _grouped_bar_chart(ax, exp_list, ["single_text", "multi_text", "dl"], "bert",
                       "BERTScore F1 — Cross Dataset (Text + DL)",
                       "BERTScore F1",
                       colors=["#1F77B4", "#FF7F0E", "#D62728"])
    plt.tight_layout()
    out_path = OUTPUT_DIR / "bertscore_cross_dataset.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    # --- DL Latency compared with single_text and multi_text ---
    fig, ax = plt.subplots(figsize=(14, 6))
    _grouped_bar_chart(ax, exp_list, ["single_text", "multi_text", "dl"], "latency",
                       "Latency (s) — Cross Dataset (Text + DL)",
                       "Latency (s)",
                       colors=["#1F77B4", "#FF7F0E", "#D62728"])
    plt.tight_layout()
    out_path = OUTPUT_DIR / "latency_cross_dataset.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# =============================================
# Main
# =============================================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 60)
    print("Collecting experiment score files...")
    print("=" * 60)
    results = collect_results()
    print(f"\nTotal experiments loaded: {len(results)}")

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # 1. Per-dataset-type metric panels (all metrics for each of single_asr, single_text, multi_asr, multi_text, dl)
    plot_category_panels(results)

    # 2. Overall score comparison
    plot_overall_score_comparison(results)

    # 3. Dedicated BERTScore
    plot_bertscore_comparison(results)

    # 4. Dedicated Latency
    plot_latency_comparison(results)

    # 5. Radar charts
    plot_radar_charts(results)

    # 6. Heatmaps
    plot_heatmaps(results)

    # 7. Grouped: Single-turn ASR vs Text, Multi-turn ASR vs Text
    plot_grouped_dataset_comparison(results)

    # 8. DL highlight + cross-dataset BERTScore / Latency
    plot_dl_highlight(results)

    # 9. Summary CSV
    write_summary_csv(results)

    print("\n" + "=" * 60)
    print(f"DONE. All charts saved to {OUTPUT_DIR.resolve()}")
    print("=" * 60)