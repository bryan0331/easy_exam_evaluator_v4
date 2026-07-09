# pipeline_single_turn.py
import json
import os
import config
import evaluators_core as core


def process_single_stream(llm_path, is_asr_stream):
    with open(config.TEST_SET_SINGLE_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    test_dict = {item['id']: item for item in test_data}
    valid_items = []
    cands, refs = [], []

    for l_item in llm_data:
        idx = l_item['id']
        if idx in test_dict:
            t_item = test_dict[idx]
            gt_ctx = [c.get("text", "") for c in t_item.get("context", []) if "text" in c]
            ret_ctx = l_item.get("retrieved_context", [])
            gt_question = t_item.get("question_text", "")
            asr_question = l_item.get("question", gt_question) if is_asr_stream else gt_question

            valid_items.append({
                "id": idx, "duration": l_item.get("duration", 0),
                "question_text": gt_question, "asr_recognition": asr_question,
                "expected_answer": t_item.get("expected_answer", ""),
                "llm_answer": l_item.get("answer", l_item.get("llm_answer", "")),
                "retrieved_context": ret_ctx, "ground_truth_context": gt_ctx
            })
            cands.append(valid_items[-1]["llm_answer"])
            refs.append(valid_items[-1]["expected_answer"])

    if not valid_items: return []

    f1_scores = core.compute_bertscore_f1(cands, refs)
    total_count = len(valid_items)

    for current_idx, item in enumerate(valid_items, start=1):
        # 💡 高亮输出当前的运行进度
        stream_label = "ASR Stream" if is_asr_stream else "Pure Text Stream"
        print(f" -> [{stream_label}] Progress: [{current_idx} / {total_count}] Evaluating Single-Turn ID: {item['id']}")

        item["wer"] = core.compute_wer(item["question_text"], item["asr_recognition"])
        item["bert_score_f1"] = f1_scores[current_idx - 1]

        l_ans_lower = item["llm_answer"].lower()
        if "outside that scope" in l_ans_lower or "only answer questions about" in l_ans_lower:
            for k in ["eval_correctness", "eval_groundedness", "eval_context_precision", "eval_context_recall",
                      "eval_retrieval_relevance", "eval_answer_relevance"]:
                item[k] = {"score": 5, "reason": "Security shield triggered, default full score."}
        else:
            item["eval_correctness"] = core.eval_correctness(item["question_text"], item["expected_answer"],
                                                             item["llm_answer"])
            item["eval_groundedness"] = core.eval_groundedness(item["llm_answer"], item["retrieved_context"])

            # 🛠️ 修复传参对齐：使用当前流所面对的实际问题 (asr_recognition) 传给评估器
            item["eval_context_precision"] = core.eval_context_precision(item["asr_recognition"],
                                                                         item["ground_truth_context"],
                                                                         item["retrieved_context"])
            item["eval_context_recall"] = core.eval_context_recall(item["ground_truth_context"],
                                                                   item["retrieved_context"])
            item["eval_retrieval_relevance"] = core.eval_retrieval_relevance_with_history(item["asr_recognition"],
                                                                                          item["ground_truth_context"],
                                                                                          item["retrieved_context"],
                                                                                          history="")
            item["eval_answer_relevance"] = core.eval_answer_relevance_with_history(item["asr_recognition"],
                                                                                    item["retrieved_context"],
                                                                                    item["llm_answer"], history="")

    return valid_items


def run_single_turn_pipeline():
    print("⏳ [Single-Turn] Launching parallel computation loops...")
    if os.path.exists(config.LLM_ASR_SINGLE_PATH):
        asr_res = process_single_stream(config.LLM_ASR_SINGLE_PATH, is_asr_stream=True)
        with open(config.OUTPUT_SINGLE_ASR_REPORT, 'w', encoding='utf-8') as f:
            json.dump(asr_res, f, ensure_ascii=False, indent=2)
        print(f"✅ Single-Turn [With ASR Stream] Done -> {config.OUTPUT_SINGLE_ASR_REPORT}")

    if os.path.exists(config.LLM_SINGLE_PATH):
        text_res = process_single_stream(config.LLM_SINGLE_PATH, is_asr_stream=False)
        with open(config.OUTPUT_SINGLE_TEXT_REPORT, 'w', encoding='utf-8') as f:
            json.dump(text_res, f, ensure_ascii=False, indent=2)
        print(f"✅ Single-Turn [Pure Text Stream] Done -> {config.OUTPUT_SINGLE_TEXT_REPORT}")


if __name__ == "__main__":
    run_single_turn_pipeline()