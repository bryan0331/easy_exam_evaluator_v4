# calculate_metrics_summary.py
import json
import os
import config

def get_mean(lst):
    v = [x for x in lst if x is not None]
    return round(sum(v) / len(v), 4) if v else 0.0

def get_stats_str(lst):
    v = [x for x in lst if x is not None]
    if not v: return "N/A"
    return f"{round((sum(v)/len(v))/1000.0, 3)}s (Max: {round(max(v)/1000.0, 3)}s, Min: {round(min(v)/1000.0, 3)}s)"

def print_stream_summary(report_path, title, is_multi=False, is_dl=False):
    if not os.path.exists(report_path):
        print(f"\n[{title}] Report file does not exist, skipping.")
        return
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    wers, berts, corrs, grounds, precs, recs, ret_rels, ans_rels = [], [], [], [], [], [], [], []
    durations = []

    if not is_multi:
        for item in data:
            if "duration" in item: durations.append(item["duration"])
            if not is_dl: wers.append(item.get("wer"))
            berts.append(item.get("bert_score_f1"))
            corrs.append(item.get("eval_correctness", {}).get("score"))
            ans_rels.append(item.get("eval_answer_relevance", {}).get("score"))
            if not is_dl:
                grounds.append(item.get("eval_groundedness", {}).get("score"))
                precs.append(item.get("eval_context_precision", {}).get("score"))
                recs.append(item.get("eval_context_recall", {}).get("score"))
                ret_rels.append(item.get("eval_retrieval_relevance", {}).get("score"))
        total = len(data)
    else:
        total = 0
        for group in data:
            if "duration" in group: durations.append(group["duration"])
            for turn in group.get("evaluated_turns", []):
                total += 1
                wers.append(turn.get("wer"))
                berts.append(turn.get("bert_score_f1"))
                corrs.append(turn.get("eval_correctness", {}).get("score"))
                grounds.append(turn.get("eval_groundedness", {}).get("score"))
                precs.append(turn.get("eval_context_precision", {}).get("score"))
                recs.append(turn.get("eval_context_recall", {}).get("score"))
                ret_rels.append(turn.get("eval_retrieval_relevance", {}).get("score"))
                ans_rels.append(turn.get("eval_answer_relevance", {}).get("score"))

    print(f"\n📊 {title} (Total Evaluated Chunks: {total})")
    if not is_dl: print(f" 🔹 Word Error Rate (WER)     : {get_mean(wers)}")
    print(f" 🔹 BERTScore F1 Similarity   : {get_mean(berts)}")
    print(f" 🔹 Answer Correctness        : {get_mean(corrs)} / 5.0")
    if not is_dl: print(f" 🔹 Groundedness (Anti-Hal)   : {get_mean(grounds)} / 5.0")
    if not is_dl: print(f" 🔹 Retrieval Relevance       : {get_mean(ret_rels)} / 5.0")
    print(f" 🔹 Answer Relevance          : {get_mean(ans_rels)} / 5.0")
    if not is_dl: print(f" 🔹 Context Precision         : {get_mean(precs)} / 5.0")
    if not is_dl: print(f" 🔹 Context Recall            : {get_mean(recs)} / 5.0")
    print(f" ⏱️  Execution Time Cost (s)   : {get_stats_str(durations)}")

def show_summary_dashboard():
    print("\n" + "=" * 25 + " 📊 RAG + ASR Joint Evaluation Summary Dashboard " + "=" * 25)
    print_stream_summary(config.OUTPUT_SINGLE_ASR_REPORT, "🟢 Single-Turn Dialogue [With ASR Audio Stream]", is_multi=False)
    print_stream_summary(config.OUTPUT_SINGLE_TEXT_REPORT, "🟢 Single-Turn Dialogue [Pure Text Standard Stream]", is_multi=False)
    print("-" * 85)
    print_stream_summary(config.OUTPUT_MULTI_ASR_REPORT, "🔵 Multi-Turn Dialogue [With ASR Audio Stream]", is_multi=True)
    print_stream_summary(config.OUTPUT_MULTI_TEXT_REPORT, "🔵 Multi-Turn Dialogue [Pure Text Standard Stream]", is_multi=True)
    print("-" * 85)
    print_stream_summary(config.OUTPUT_DL_EXAM_REPORT, "🔥 Deep Learning Exam [Pure Text No-RAG Stream]", is_multi=False, is_dl=True)
    print("\n" + "=" * 98)

if __name__ == "__main__":
    show_summary_dashboard()