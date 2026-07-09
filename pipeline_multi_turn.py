# pipeline_multi_turn.py
import json
import os
import config
import evaluators_core as core


def process_multi_stream(llm_path, is_asr_stream):
    with open(config.TEST_SET_MULTI_PATH, "r", encoding="utf-8") as f:
        test_multi = json.load(f)
    with open(llm_path, "r", encoding="utf-8") as f:
        llm_multi_data = json.load(f)

    llm_multi_dict = {item["id"]: item for item in llm_multi_data}
    test_dict = {item["id"]: item for item in test_multi}

    all_cands, all_refs = [], []
    flat_turn_pointers = []
    final_output = []

    for idx, l_group in llm_multi_dict.items():
        if idx in test_dict:
            t_group = test_dict[idx]
            g_ctx = [c.get("text", "") for c in t_group.get("context", []) if "text" in c]

            output_group = {
                "id": idx, "duration": l_group.get("duration", 0), "evaluated_turns": []
            }
            history_accumulator = []
            l_turns = l_group.get("turns", [])
            t_turns = t_group.get("turns", [])

            for i, l_turn in enumerate(l_turns):
                user_idx, asst_idx = i * 2, i * 2 + 1
                if asst_idx >= len(t_turns): break

                user_turn = t_turns[user_idx]
                assistant_turn = t_turns[asst_idx]

                gt_question = user_turn.get("question_text", "")
                asr_question = l_turn.get("question", gt_question) if is_asr_stream else gt_question
                e_ans = assistant_turn.get("expected_answer", "")
                l_ans = l_turn.get("answer", "")
                r_ctx = l_turn.get("retrieved_context", [])

                history_str = "\n".join(history_accumulator) if history_accumulator else "No dialogue history."

                turn_report = {
                    "turn_index": i + 1, "question_text": gt_question, "asr_recognition": asr_question,
                    "expected_answer": e_ans, "llm_answer": l_ans,
                    "retrieved_context": r_ctx, "ground_truth_context": g_ctx, "history_context": history_str
                }
                output_group["evaluated_turns"].append(turn_report)
                all_cands.append(l_ans)
                all_refs.append(e_ans)
                flat_turn_pointers.append(turn_report)

                history_accumulator.append(f"User: {asr_question}")
                history_accumulator.append(f"Assistant: {l_ans}")

            final_output.append(output_group)

    if not all_cands: return []

    f1_scores = core.compute_bertscore_f1(all_cands, all_refs)
    total_count = len(flat_turn_pointers)

    for current_idx, turn in enumerate(flat_turn_pointers, start=1):
        stream_label = "ASR Stream" if is_asr_stream else "Pure Text Stream"
        print(
            f" -> [{stream_label}] Progress: [{current_idx} / {total_count}] Evaluating Multi-Turn Turn {turn['turn_index']}...")

        turn["wer"] = core.compute_wer(turn["question_text"], turn["asr_recognition"])
        turn["bert_score_f1"] = f1_scores[current_idx - 1]

        l_ans_lower = turn["llm_answer"].lower()
        if "outside that scope" in l_ans_lower or "only answer questions about" in l_ans_lower:
            for k in ["eval_correctness", "eval_groundedness", "eval_context_precision", "eval_context_recall",
                      "eval_retrieval_relevance", "eval_answer_relevance"]:
                turn[k] = {"score": 5, "reason": "Security shield triggered."}
        else:
            # 🛠️ 严格修正传参映射，确保与 evaluators_core.py 完全一致
            turn["eval_correctness"] = core.eval_correctness(turn["question_text"], turn["expected_answer"],
                                                             turn["llm_answer"])
            turn["eval_groundedness"] = core.eval_groundedness(turn["llm_answer"], turn["retrieved_context"])

            turn["eval_context_precision"] = core.eval_context_precision(turn["asr_recognition"],
                                                                         turn["ground_truth_context"],
                                                                         turn["retrieved_context"])
            turn["eval_context_recall"] = core.eval_context_recall(turn["ground_truth_context"],
                                                                   turn["retrieved_context"])

            # 🛠️ 修复此处多余或错位的传参 Bug
            turn["eval_retrieval_relevance"] = core.eval_retrieval_relevance_with_history(
                turn["asr_recognition"],
                turn["ground_truth_context"],
                turn["retrieved_context"],
                history=turn["history_context"]
            )
            turn["eval_answer_relevance"] = core.eval_answer_relevance_with_history(
                turn["asr_recognition"],
                turn["retrieved_context"],
                turn["llm_answer"],
                history=turn["history_context"]
            )

    return final_output


def run_multi_turn_pipeline():
    print("⏳ [Multi-Turn] Launching parallel computation loops...")
    if os.path.exists(config.LLM_ASR_MULTI_PATH):
        asr_res = process_multi_stream(config.LLM_ASR_MULTI_PATH, is_asr_stream=True)
        with open(config.OUTPUT_MULTI_ASR_REPORT, "w", encoding="utf-8") as f:
            json.dump(asr_res, f, ensure_ascii=False, indent=2)
        print(f"✅ Multi-Turn [With ASR Stream] Done -> {config.OUTPUT_MULTI_ASR_REPORT}")

    if os.path.exists(config.LLM_MULTI_PATH):
        text_res = process_multi_stream(config.LLM_MULTI_PATH, is_asr_stream=False)
        with open(config.OUTPUT_MULTI_TEXT_REPORT, "w", encoding="utf-8") as f:
            json.dump(text_res, f, ensure_ascii=False, indent=2)
        print(f"✅ Multi-Turn [Pure Text Stream] Done -> {config.OUTPUT_MULTI_TEXT_REPORT}")


if __name__ == "__main__":
    run_multi_turn_pipeline()