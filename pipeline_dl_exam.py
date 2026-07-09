# pipeline_dl_exam.py
import json
import os
import config
import evaluators_core as core


def run_dl_exam_pipeline():
    print("⏳ [DL-Exam] Loading Deep Learning dataset for alignment...")

    if not os.path.exists(config.LLM_DL_EXAM_PATH) or not os.path.exists(config.TEST_SET_DL_EXAM_PATH):
        print("❌ Essential dataset files missing.")
        return

    with open(config.LLM_DL_EXAM_PATH, "r", encoding="utf-8") as f:
        llm_data = json.load(f)
    with open(config.TEST_SET_DL_EXAM_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_dict = {item["id"]: item for item in test_data}
    valid_items = []
    cands, refs = [], []

    for l_item in llm_data:
        idx = l_item["id"]
        if idx in test_dict:
            t_item = test_dict[idx]
            q_text = t_item.get("question_text", l_item.get("question", ""))
            llm_ans = l_item.get("answer", l_item.get("llm_answer", ""))

            valid_items.append({
                "id": idx, "duration": l_item.get("duration", 0), "question_text": q_text,
                "expected_answer": t_item.get("expected_answer", ""), "llm_answer": llm_ans
            })
            cands.append(llm_ans)
            refs.append(valid_items[-1]["expected_answer"])

    if not valid_items: return

    f1_scores = core.compute_bertscore_f1(cands, refs)
    total_count = len(valid_items)

    for current_idx, item in enumerate(valid_items, start=1):
        # 💡 高亮输出当前的运行进度
        print(f" -> [DL-Exam] Progress: [{current_idx} / {total_count}] Evaluating Exam ID: {item['id']}")
        item["bert_score_f1"] = f1_scores[current_idx - 1]

        item["eval_correctness"] = core.eval_correctness(item["question_text"], item["expected_answer"],
                                                         item["llm_answer"])
        item["eval_answer_relevance"] = core.eval_answer_relevance_with_history(item["question_text"], [],
                                                                                item["llm_answer"], history="")

    os.makedirs(os.path.dirname(config.OUTPUT_DL_EXAM_REPORT), exist_ok=True)
    with open(config.OUTPUT_DL_EXAM_REPORT, "w", encoding="utf-8") as f:
        json.dump(valid_items, f, ensure_ascii=False, indent=2)
    print(f"🎉 [DL-Exam] Complete -> {config.OUTPUT_DL_EXAM_REPORT}")


if __name__ == "__main__":
    run_dl_exam_pipeline()