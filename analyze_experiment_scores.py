import re
from pathlib import Path
from typing import Dict, List, Tuple
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOTS = [Path("baseline"), Path("chunk_page"), Path("final_rewrite_hyde")]
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

CATEGORY_LABELS = {
    "single_asr": "Single-Turn with ASR",
    "single_text": "Single-Turn Pure Text",
    "multi_asr": "Multi-Turn with ASR",
    "multi_text": "Multi-Turn Pure Text",
    "dl": "DL Exam",
}

METRICS = [
    ("bert", "BERTScore F1"),
    ("correctness", "Answer Correctness"),
    ("groundedness", "Groundedness"),
    ("retrieval_relevance", "Retrieval Relevance"),
    ("answer_relevance", "Answer Relevance"),
    ("context_precision", "Context Precision"),
    ("context_recall", "Context Recall"),
    ("latency", "Latency (s)"),
]

SECTION_PATTERNS = {
    "single_asr": "Single-Turn Dialogue [With ASR Audio Stream]",
    "single_text": "Single-Turn Dialogue [Pure Text Standard Stream]",
    "multi_asr": "Multi-Turn Dialogue [With ASR Audio Stream]",
    "multi_text": "Multi-Turn Dialogue [Pure Text Standard Stream]",
    "dl": "Deep Learning Exam [Pure Text No-RAG Stream]",
}


def find_score_files() -> List[Path]:
    files = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() in {"scores", "score"}:
                files.append(path)
    return sorted(files)


def parse_metric_value(section_text: str, metric_name: str) -> float:
    if metric_name == "latency":
        m = re.search(r"Execution Time Cost \(s\)\s*:\s*([0-9.]+)s", section_text)
        if m:
            return float(m.group(1))
        return math.nan

    patterns = {
        "bert": r"BERTScore F1 Similarity\s*:\s*([0-9.]+)",
        "correctness": r"Answer Correctness\s*:\s*([0-9.]+)",
        "groundedness": r"Groundedness \(Anti-Hal\)\s*:\s*([0-9.]+)",
        "retrieval_relevance": r"Retrieval Relevance\s*:\s*([0-9.]+)",
        "answer_relevance": r"Answer Relevance\s*:\s*([0-9.]+)",
        "context_precision": r"Context Precision\s*:\s*([0-9.]+)",
        "context_recall": r"Context Recall\s*:\s*([0-9.]+)",
    }
    m = re.search(patterns[metric_name], section_text)
    if m:
        return float(m.group(1))
    return math.nan


def parse_score_file(path: Path) -> Dict[str, Dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    data: Dict[str, Dict[str, float]] = {}
    for category, marker in SECTION_PATTERNS.items():
        if marker not in text:
            continue
        # split by the marker lines
        start = text.find(marker)
        if start == -1:
            continue
        end = len(text)
        for next_marker in SECTION_PATTERNS.values():
            if next_marker != marker:
                nxt = text.find(next_marker, start + 1)
                if nxt != -1 and nxt < end:
                    end = nxt
        section_text = text[start:end]
        metrics = {metric: parse_metric_value(section_text, metric_name) for metric, metric_name in [(m, m) for m in ["bert", "correctness", "groundedness", "retrieval_relevance", "answer_relevance", "context_precision", "context_recall", "latency"]]}
        data[category] = metrics
    return data


def get_experiment_name(path: Path) -> str:
    return path.parent.as_posix()


def collect_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    results = {}
    for path in find_score_files():
        exp_name = get_experiment_name(path)
        try:
            parsed = parse_score_file(path)
        except Exception as exc:
            print(f"Skip {path}: {exc}")
            continue
        if parsed:
            results[exp_name] = parsed
    return results


def plot_metric_comparison(results: Dict[str, Dict[str, Dict[str, float]]], category: str, metric: str, title: str, output_path: Path):
    labels = []
    values = []
    for exp_name in sorted(results):
        values_for_exp = results[exp_name].get(category, {})
        if metric in values_for_exp and not math.isnan(values_for_exp[metric]):
            labels.append(exp_name)
            values.append(values_for_exp[metric])
    if not labels:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#4C78A8")
    plt.xticks(rotation=30, ha="right")
    plt.title(title, fontsize=14)
    plt.ylabel(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_multi_metric_panel(results: Dict[str, Dict[str, Dict[str, float]]], category: str, title: str, output_path: Path, metrics: List[Tuple[str, str]]):
    experiments = sorted(results)
    if not experiments:
        return
    # only keep experiments with at least one value for the selected metrics
    valid_exps = [exp for exp in experiments if any(metric in results[exp].get(category, {}) and not math.isnan(results[exp][category][metric]) for metric, _ in metrics)]
    if not valid_exps:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        values = []
        labels = []
        for exp in valid_exps:
            val = results[exp].get(category, {}).get(metric_key)
            if val is None or math.isnan(val):
                continue
            labels.append(exp)
            values.append(val)
        if labels:
            ax.bar(labels, values, color="#4C78A8")
            ax.set_title(metric_label)
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=30)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_latency_comparison(results: Dict[str, Dict[str, Dict[str, float]]], output_path: Path):
    categories = ["single_asr", "single_text", "multi_asr", "multi_text", "dl"]
    labels = []
    values = []
    for exp in sorted(results):
        for category in categories:
            val = results[exp].get(category, {}).get("latency")
            if val is None or math.isnan(val):
                continue
            labels.append(f"{exp} | {CATEGORY_LABELS[category]}")
            values.append(val)
    if not labels:
        return
    plt.figure(figsize=(14, 7))
    plt.bar(labels, values, color="#F58518")
    plt.xticks(rotation=45, ha="right")
    plt.title("Latency Comparison by Experiment and Dataset Type")
    plt.ylabel("Latency (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_overall_score(metrics: Dict[str, float], kind: str) -> float:
    if kind == "dl":
        keys = ["bert", "correctness", "answer_relevance"]
    else:
        keys = ["bert", "correctness", "groundedness", "retrieval_relevance", "answer_relevance", "context_precision", "context_recall"]
    values = []
    for key in keys:
        v = metrics.get(key)
        if v is not None and not math.isnan(v):
            values.append(float(v))
    return round(sum(values) / len(values), 3) if values else math.nan


def plot_overall_score_comparison(results: Dict[str, Dict[str, Dict[str, float]]], output_path: Path):
    categories = [
        ("single_asr", "Single-Turn with ASR"),
        ("single_text", "Single-Turn Pure Text"),
        ("multi_asr", "Multi-Turn with ASR"),
        ("multi_text", "Multi-Turn Pure Text"),
        ("dl", "DL Exam"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    task_groups = [
        (["single_asr", "single_text"], "Single-Turn Overall Score"),
        (["multi_asr", "multi_text"], "Multi-Turn Overall Score"),
        (["dl"], "DL Exam Overall Score"),
    ]

    for ax, (group, title) in zip(axes, task_groups):
        labels = []
        values = []
        for exp in sorted(results):
            for category_key in group:
                metrics = results[exp].get(category_key, {})
                if not metrics:
                    continue
                score = compute_overall_score(metrics, category_key if category_key == "dl" else "other")
                if not math.isnan(score):
                    labels.append(f"{exp} | {CATEGORY_LABELS[category_key]}")
                    values.append(score)
        if labels:
            ax.bar(labels, values, color="#2CA02C")
            ax.set_title(title)
            ax.set_ylabel("Overall Composite Score")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    results = collect_results()
    print(f"Loaded {len(results)} experiment score files")
    for exp_name, categories in sorted(results.items()):
        print(exp_name, sorted(categories.keys()))

    # Single-turn metrics
    plot_multi_metric_panel(
        results,
        "single_asr",
        "Single-Turn with ASR: Core Metrics",
        OUTPUT_DIR / "single_turn_asr_metrics.png",
        [("bert", "BERTScore F1"), ("correctness", "Answer Correctness"), ("answer_relevance", "Answer Relevance"), ("context_precision", "Context Precision")],
    )
    plot_multi_metric_panel(
        results,
        "single_text",
        "Single-Turn Pure Text: Core Metrics",
        OUTPUT_DIR / "single_turn_text_metrics.png",
        [("bert", "BERTScore F1"), ("correctness", "Answer Correctness"), ("answer_relevance", "Answer Relevance"), ("context_precision", "Context Precision")],
    )

    # Multi-turn metrics
    plot_multi_metric_panel(
        results,
        "multi_asr",
        "Multi-Turn with ASR: Core Metrics",
        OUTPUT_DIR / "multi_turn_asr_metrics.png",
        [("bert", "BERTScore F1"), ("correctness", "Answer Correctness"), ("answer_relevance", "Answer Relevance"), ("context_precision", "Context Precision")],
    )
    plot_multi_metric_panel(
        results,
        "multi_text",
        "Multi-Turn Pure Text: Core Metrics",
        OUTPUT_DIR / "multi_turn_text_metrics.png",
        [("bert", "BERTScore F1"), ("correctness", "Answer Correctness"), ("answer_relevance", "Answer Relevance"), ("context_precision", "Context Precision")],
    )

    # DL exam metrics
    plot_multi_metric_panel(
        results,
        "dl",
        "DL Exam: Core Metrics",
        OUTPUT_DIR / "dl_exam_metrics.png",
        [("bert", "BERTScore F1"), ("correctness", "Answer Correctness"), ("answer_relevance", "Answer Relevance"), ("latency", "Latency")],
    )

    # Latency comparison
    plot_latency_comparison(results, OUTPUT_DIR / "latency_comparison.png")

    # Overall composite score comparison
    plot_overall_score_comparison(results, OUTPUT_DIR / "overall_score_comparison.png")

    print(f"Saved charts to {OUTPUT_DIR.resolve()}")
