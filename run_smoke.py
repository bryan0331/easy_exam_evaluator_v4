# run_smoke_baseline.py
import json
import os
import config
import evaluators_core as core


def load_json_safe(path):
    abs_path = os.path.abspath(path)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f" ❌ Error reading JSON from {abs_path}: {e}")
            return None
    print(f" ⚠️ File not found at absolute path: {abs_path}")
    return None


# 💡 已经去掉 test_ 前缀，变为纯业务函数名，彻底摘除“单元测试用例”标签
def run_single_turn_stream(is_asr=True, limit=1):
    stream_label = "With ASR Stream" if is_asr else "Pure Text Stream"
    llm_path = config.LLM_ASR_SINGLE_PATH if is_asr else config.LLM_SINGLE_PATH
    out_path = config.OUTPUT_SINGLE_ASR_REPORT if is_asr else config.OUTPUT_SINGLE_TEXT_REPORT

    print(f"\n🔄 [Single-Turn] >>> 正在装载流: {stream_label} <<<")
    llm_data = load_json_safe(llm_path)
    test_data = load_json_safe(config.TEST_SET_SINGLE_PATH)

    if not llm_data or not test_data:
        print(f" ⏩ Skipping Single-Turn [{stream_label}] due to missing or empty source datasets.")
        return []

    test_dict = {item['id']: item for item in test_data}
    results = []

    subset = llm_data[:limit]
    total_count = len(subset)
    cands = [l.get("answer", l.get("llm_answer", "")) for l in subset if l['id'] in test_dict]
    refs = [test_dict[l['id']].get("expected_answer", "") for l in subset if l['id'] in test_dict]
    f1_scores = core.compute_bertscore_f1(cands, refs) if cands else []

    idx_f1 = 0
    for current_idx, l_item in enumerate(subset, start=1):
        idx = l_item['id']
        if idx in test_dict:
            t_item = test_dict[idx]
            gt_q = t_item.get("question_text", "")
            asr_q = l_item.get("question", gt_q) if is_asr else gt_q
            llm_ans = l_item.get("answer", l_item.get("llm_answer", ""))
            ret_ctx = l_item.get("retrieved_context", [])
            gt_ctx = [c.get("text", "") for c in t_item.get("context", []) if "text" in c]

            print(f" ⏳ [{stream_label}] [{current_idx} / {total_count}] Processing Token Metrics for ID: {idx}...")

            correctness = core.eval_correctness(gt_q, t_item["expected_answer"], llm_ans)
            groundedness = core.eval_groundedness(llm_ans, ret_ctx)

            precision = core.eval_context_precision(asr_q, gt_ctx, ret_ctx)
            recall = core.eval_context_recall(gt_ctx, ret_ctx)
            ret_rel = core.eval_retrieval_relevance_with_history(asr_q, gt_ctx, ret_ctx, history="")
            ans_rel = core.eval_answer_relevance_with_history(asr_q, ret_ctx, llm_ans, history="")

            results.append({
                "id": idx, "duration": l_item.get("duration", 0),
                "question_text": gt_q, "asr_recognition": asr_q,
                "expected_answer": t_item["expected_answer"], "llm_answer": llm_ans,
                "wer": core.compute_wer(gt_q, asr_q),
                "bert_score_f1": f1_scores[idx_f1] if idx_f1 < len(f1_scores) else 0.0,
                "eval_correctness": correctness, "eval_groundedness": groundedness,
                "eval_context_precision": precision, "eval_context_recall": recall,
                "eval_retrieval_relevance": ret_rel, "eval_answer_relevance": ans_rel
            })
            idx_f1 += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f" 💾 Evaluation result successfully exported to: {out_path}")
    return results


# 💡 已经去掉 test_ 前缀
def run_multi_turn_stream(is_asr=True, limit=1):
    stream_label = "With ASR Stream" if is_asr else "Pure Text Stream"
    llm_path = config.LLM_ASR_MULTI_PATH if is_asr else config.LLM_MULTI_PATH
    out_path = config.OUTPUT_MULTI_ASR_REPORT if is_asr else config.OUTPUT_MULTI_TEXT_REPORT

    print(f"\n🔄 [Multi-Turn] >>> 正在装载流: {stream_label} <<<")
    llm_data = load_json_safe(llm_path)
    test_data = load_json_safe(config.TEST_SET_MULTI_PATH)

    if not llm_data or not test_data:
        print(f" ⏩ Skipping Multi-Turn [{stream_label}] due to missing or empty source datasets.")
        return []

    test_dict = {item['id']: item for item in test_data}
    results = []

    subset = llm_data[:limit]
    total_count = len(subset)
    for current_idx, l_group in enumerate(subset, start=1):
        idx = l_group['id']
        if idx in test_dict:
            t_group = test_dict[idx]
            group_report = {"id": idx, "duration": l_group.get("duration", 0), "evaluated_turns": []}

            history_accumulator = []
            g_ctx = [c.get("text", "") for c in t_group.get("context", []) if "text" in c]
            l_turns = l_group.get("turns", [])
            t_turns = t_group.get("turns", [])

            for i, l_turn in enumerate(l_turns[:1]):
                if (i * 2 + 1) >= len(t_turns): break
                user_turn = t_turns[i * 2]
                assistant_turn = t_turns[i * 2 + 1]

                gt_q = user_turn.get("question_text", "")
                asr_q = l_turn.get("question", gt_q) if is_asr else gt_q
                llm_ans = l_turn.get("answer", "")
                r_ctx = l_turn.get("retrieved_context", [])
                e_ans = assistant_turn.get("expected_answer", "")

                history_str = "\n".join(history_accumulator) if history_accumulator else "No dialogue history."

                print(f" ⏳ [{stream_label}] [{current_idx} / {total_count}] Processing Token Metrics for Session: {idx} Turn {i + 1}...")

                correctness = core.eval_correctness(gt_q, e_ans, llm_ans)
                groundedness = core.eval_groundedness(llm_ans, r_ctx)

                precision = core.eval_context_precision(asr_q, g_ctx, r_ctx)
                recall = core.eval_context_recall(g_ctx, r_ctx)
                ret_rel = core.eval_retrieval_relevance_with_history(asr_q, g_ctx, r_ctx, history=history_str)
                ans_rel = core.eval_answer_relevance_with_history(asr_q, r_ctx, llm_ans, history=history_str)

                turn_f1 = core.compute_bertscore_f1([llm_ans], [e_ans])[0]

                group_report["evaluated_turns"].append({
                    "turn_index": i + 1, "question_text": gt_q, "asr_recognition": asr_q,
                    "expected_answer": e_ans, "llm_answer": llm_ans,
                    "wer": core.compute_wer(gt_q, asr_q), "bert_score_f1": turn_f1,
                    "eval_correctness": correctness, "eval_groundedness": groundedness,
                    "eval_context_precision": precision, "eval_context_recall": recall,
                    "eval_retrieval_relevance": ret_rel, "eval_answer_relevance": ans_rel
                })
                history_accumulator.append(f"User: {asr_q}")
                history_accumulator.append(f"Assistant: {llm_ans}")
            results.append(group_report)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f" 💾 Evaluation result successfully exported to: {out_path}")
    return results


# 💡 已经去掉 test_ 前缀
def run_dl_exam_stream(limit=1):
    print(f"\n🔄 [DL-Exam Stream] >>> 正在装载深度学习考试流 <<<")
    llm_data = load_json_safe(config.LLM_DL_EXAM_PATH)
    test_data = load_json_safe(config.TEST_SET_DL_EXAM_PATH)
    if not llm_data or not test_data:
        print(f" ⏩ Skipping DL-Exam Stream due to missing or empty input data layers.")
        return []
    test_dict = {item['id']: item for item in test_data}
    results = []

    subset = llm_data[:limit]
    total_count = len(subset)
    cands = [l.get("answer", l.get("llm_answer", "")) for l in subset if l['id'] in test_dict]
    refs = [test_dict[l['id']].get("expected_answer", "") for l in subset if l['id'] in test_dict]
    f1_scores = core.compute_bertscore_f1(cands, refs) if cands else []

    idx_f1 = 0
    for current_idx, l_item in enumerate(subset, start=1):
        idx = l_item['id']
        if idx in test_dict:
            t_item = test_dict[idx]
            gt_q = t_item.get("question_text", "")
            llm_ans = l_item.get("answer", l_item.get("llm_answer", ""))

            print(f" ⏳ [DL-Exam Stream] [{current_idx} / {total_count}] Processing Exam ID: {idx}...")
            correctness = core.eval_correctness(gt_q, t_item.get("expected_answer", ""), llm_ans)
            ans_rel = core.eval_answer_relevance_with_history(gt_q, [], llm_ans, history="")

            results.append({
                "id": idx, "duration": l_item.get("duration", 0), "wer": 0.0,
                "question_text": gt_q, "expected_answer": t_item.get("expected_answer", ""), "llm_answer": llm_ans,
                "bert_score_f1": f1_scores[idx_f1] if idx_f1 < len(f1_scores) else 0.0,
                "eval_correctness": correctness, "eval_answer_relevance": ans_rel
            })
            idx_f1 += 1

    os.makedirs(os.path.dirname(config.OUTPUT_DL_EXAM_REPORT), exist_ok=True)
    with open(config.OUTPUT_DL_EXAM_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f" 💾 Evaluation result successfully exported to: {config.OUTPUT_DL_EXAM_REPORT}")
    return results


def print_smoke_summary(data, title, is_multi=False, is_dl=False):
    def get_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    wers, berts, corrs, grounds, precs, recs, ret_rels, ans_rels = [], [], [], [], [], [], [], []
    durations = []

    if not data:
        print(f"\n📊 {title} (Smoke Test Chunks: 0) -> No data processed or file skipped.")
        return

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

    mean_dur = f"{round((sum(durations) / len(durations)) / 1000.0, 3)}s" if durations else "N/A"
    print(f"\n📊 {title} (Smoke Test Chunks: {total})")
    if not is_dl: print(f" 🔹 Word Error Rate (WER)     : {get_mean(wers)}")
    print(f" 🔹 BERTScore F1 Similarity   : {get_mean(berts)}")
    print(f" 🔹 Answer Correctness        : {get_mean(corrs)} / 5.0")
    if not is_dl: print(f" 🔹 Groundedness (Anti-Hal)   : {get_mean(grounds)} / 5.0")
    if not is_dl: print(f" 🔹 Retrieval Relevance       : {get_mean(ret_rels)} / 5.0")
    print(f" 🔹 Answer Relevance          : {get_mean(ans_rels)} / 5.0")
    if not is_dl: print(f" 🔹 Context Precision         : {get_mean(precs)} / 5.0")
    if not is_dl: print(f" 🔹 Context Recall            : {get_mean(recs)} / 5.0")
    print(f" ⏱️  Average Execution Time    : {mean_dur}")


if __name__ == "__main__":
    print("\n" + "=" * 22 + " 🧪 Executing Smoke Test Base Logic [Real Computation Mode] " + "=" * 22)

    # 💡 核心对齐：主入口里完全按照 5 条独立流逻辑，依次触发拉取
    s_asr = run_single_turn_stream(is_asr=True, limit=1)
    s_txt = run_single_turn_stream(is_asr=False, limit=1)  # 👈 这次 Pure Text Stream 会被强制触发！
    m_asr = run_multi_turn_stream(is_asr=True, limit=1)
    m_txt = run_multi_turn_stream(is_asr=False, limit=1)   # 👈 这次 Pure Text Stream 会被强制触发！
    dl_exam = run_dl_exam_stream(limit=1)

    print("\n" + "=" * 25 + " 📊 Terminal Instant Output: Multi-Stream Baseline Billboard " + "=" * 25)
    print_smoke_summary(s_asr, "🟢 Single-Turn Dialogue [With ASR Audio Stream]", is_multi=False)
    print_smoke_summary(s_txt, "🟢 Single-Turn Dialogue [Pure Text Standard Stream]", is_multi=False)
    print("-" * 85)
    print_smoke_summary(m_asr, "🔵 Multi-Turn Dialogue [With ASR Audio Stream]", is_multi=True)
    print_smoke_summary(m_txt, "🔵 Multi-Turn Dialogue [Pure Text Standard Stream]", is_multi=True)
    print("-" * 85)
    print_smoke_summary(dl_exam, "🔥 Deep Learning Exam [Pure Text No-RAG Stream]", is_multi=False, is_dl=True)