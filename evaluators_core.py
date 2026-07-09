# evaluators_core.py
import json
import re
from jiwer import wer
from bert_score import score
from utils_llm import query_astron

# ==================== 1. ASR Word Error Rate ====================\

def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    if not reference_text or not hypothesis_text:
        return 1.0 if reference_text != hypothesis_text else 0.0
    try:
        return float(wer(reference_text.strip(), hypothesis_text.strip()))
    except Exception as e:
        print(f"   ⚠️ Error computing WER: {e}")
        return 1.0

# ==================== 2. BERTScore Similarity ====================\

def compute_bertscore_f1(cands: list, refs: list) -> list:
    if not cands or not refs:
        return [0.0] * len(cands)
    try:
        P, R, F1 = score(cands, refs, lang="en", verbose=False)
        return [float(x) for x in F1]
    except Exception as e:
        print(f"   ⚠️ Error computing BERTScore: {e}")
        return [0.0] * len(cands)

# ==================== 3. JSON Parser Recovery Block ====================\

def _parse_llm_json(res_text: str) -> dict:
    if not res_text:
        return {"score": 1, "reason": "Empty response from LLM."}
    try:
        # 清理可能带有的 Markdown 代码块包裹
        cleaned_text = re.sub(r'^```json\s*', '', res_text.strip(), flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
        return json.loads(cleaned_text)
    except Exception as e:
        # 🧪 增强型非闭合残缺 JSON 暴力抢救提取逻辑
        try:
            score_match = re.search(r'"score"\s*:\s*(\d+)', res_text)
            reason_match = re.search(r'"reason"\s*:\s*"(.*)', res_text, re.DOTALL)
            if score_match:
                score_val = int(score_match.group(1))
                reason_val = reason_match.group(1).strip() if reason_match else "Reason text truncated."
                # 递归清除尾部残缺的引号、逗号、括号和空白符
                reason_val = re.sub(r'["\}\s,\]]+$', '', reason_val)
                return {"score": score_val, "reason": reason_val}
        except:
            pass
        return {"score": 1, "reason": f"Failed to parse JSON. Raw text block: {res_text[:120]}"}

# ==================== 4. Core Universal RAG Evaluators (Single & Multi-Turn) ====================\

def eval_correctness(gt_question: str, expected_answer: str, llm_answer: str) -> dict:
    system_prompt = "You are an expert academic evaluator grading university student exam papers."
    user_prompt = f"""Compare the student's answer (LLM Answer) against the model answer (Expected Answer) to judge the exact correctness for the question.

[Question]: {gt_question}
[Expected Answer]: {expected_answer}
[LLM Answer]: {llm_answer}

Grading Schema (1-5 Scale):
5 - Fully Correct: Completely accurate, covers all key points of the expected answer perfectly.
4 - Mostly Correct: Covers main points but misses trivial details, or contains tiny, non-critical inaccuracies.
3 - Partially Correct: Hits some core concepts but leaves out critical pieces or contains noticeable flaws.
2 - Minor Alignment: Mentions relevant concepts but fails to answer the actual intent correctly.
1 - Totally Incorrect: Completely wrong, irrelevant, or hallucinated content.

Output requirements: Return a valid JSON object with exactly two keys:
{{
  "score": <integer from 1 to 5>,
  "reason": "<a brief, concise explanation in English under 50 words>"
}}"""
    return _parse_llm_json(query_astron(system_prompt, user_prompt))

def eval_groundedness(llm_answer: str, contexts: list) -> dict:
    ctx_str = "\n".join([f"Context Block {i+1}: {c}" for i, c in enumerate(contexts)]) if contexts else "No context available."
    system_prompt = "You are an expert factual checker evaluating hallucinations in RAG systems."
    user_prompt = f"""Analyze the provided retrieved context blocks and verify whether the system response claims are strictly derived from them.

[Retrieved Contexts]:
{ctx_str}
[System Actual Response]:
{llm_answer}

Grading Schema (1-5 Scale):
5 - Perfect: No hallucination. Every claim in the response is directly supported by the context.
4 - Good: Tiny extrapolation. The response is almost entirely grounded but adds minor harmless common-sense filler.
3 - Medial: Noticeable ungrounded claims. Contains statements that cannot be verified by the text, but doesn't actively contradict facts.
2 - Poor: Heavy hallucination. Major technical facts in the response are fabricated out of thin air.
1 - Dangerous: Actively contradicts or completely ignores the provided contexts.

Output requirements: Return a valid JSON object with exactly two keys:
{{
  "score": <integer from 1 to 5>,
  "reason": "<a brief, concise explanation in English under 50 words>"
}}"""
    return _parse_llm_json(query_astron(system_prompt, user_prompt))

def eval_context_precision(question_used: str, gt_contexts: list, ret_contexts: list) -> dict:
    ctx_str = "\n".join([f"Retrieved Context Block {i+1}: {c}" for i, c in enumerate(ret_contexts)]) if ret_contexts else "No retrieved context available."
    gt_str = "\n".join([f"Ground Truth Reference {i+1}: {g}" for i, g in enumerate(gt_contexts)]) if gt_contexts else "No ground truth context available."
    system_prompt = "You are an information retrieval specialist judging RAG context relevance."
    user_prompt = f"""Evaluate whether the retrieved context chunks are highly relevant, precise, and necessary to form the expected ground truth answer.

[Target Question]: {question_used}
[Ground Truth References]:
{gt_str}
[Retrieved Context Chunks]:
{ctx_str}

Grading Schema (1-5 Scale):
5 - Highly Precise: Every chunk contains highly relevant, non-redundant information needed to solve the question.
4 - Good: Most chunks are relevant, with very minor irrelevant text blocks.
3 - Average: Half of the context blocks are filler or unrelated to the specific question details.
2 - Low Precision: Only a small fragment in one chunk is useful; the rest is noisy/unrelated.
1 - Irrelevant: None of the chunks provide any actionable information to answer the question.

Output requirements: Return a valid JSON object with exactly two keys:
{{
  "score": <integer from 1 to 5>,
  "reason": "<a brief, concise explanation in English under 50 words>"
}}"""
    return _parse_llm_json(query_astron(system_prompt, user_prompt))

def eval_context_recall(gt_contexts: list, ret_contexts: list) -> dict:
    ctx_str = "\n".join([f"Retrieved Context Block {i+1}: {c}" for i, c in enumerate(ret_contexts)]) if ret_contexts else "No retrieved context available."
    gt_str = "\n".join([f"Ground Truth Reference {i+1}: {g}" for i, g in enumerate(gt_contexts)]) if gt_contexts else "No ground truth context available."
    system_prompt = "You are an expert QA validator measuring context recall coverage."
    user_prompt = f"""Check if the retrieved context blocks contain ALL the necessary background facts listed in the expected ground truth context.

[Ground Truth References]:
{gt_str}
[Retrieved Context Chunks]:
{ctx_str}

Grading Schema (1-5 Scale):
5 - Complete Recall: 100% of the facts required by the ground truth are successfully captured in the retrieved context blocks.
4 - High Recall: Covers 75-90% of the required facts, missing just one minor sub-detail.
3 - Partial Recall: Roughly half of the required facts can be found; significant knowledge gaps exist.
2 - Poor Recall: Most facts are missing, only capturing isolated key terms.
1 - Zero Recall: Absolutely none of the required facts from the model answer are present.

Output requirements: Return a valid JSON object with exactly two keys:
{{
  "score": <integer from 1 to 5>,
  "reason": "<a brief, concise explanation in English under 50 words>"
}}"""
    return _parse_llm_json(query_astron(system_prompt, user_prompt))

def eval_retrieval_relevance_with_history(question_used: str, gt_contexts: list, ret_contexts: list, history: str = "") -> dict:
    ctx_str = "\n".join([f"Retrieved Context Block {i+1}: {c}" for i, c in enumerate(ret_contexts)]) if ret_contexts else "No retrieved context available."
    system_prompt = "You are an advanced search quality rater."
    user_prompt = f"""Evaluate the raw topical relevance of the retrieved contexts specifically to the intent of the question.

[Dialogue History]:
{history if history else "None"}
[Target Question]: {question_used}
[Retrieved Contexts]:
{ctx_str}

Grading Schema (1-5 Scale):
5 - Perfect Relevance: Directly and cleanly targets the specific question entity, context, and domain.
4 - Relevant: Closely related but contains some overly broad information.
3 - Marginally Relevant: Touches upon the general topic but misses the specific angle of the user prompt.
2 - Weakly Relevant: Contains keyword matching overlap but completely misses the core semantic intent.
1 - Totally Irrelevant: Off-topic background noises.

Output requirements: Return a valid JSON object with exactly two keys:
{{
  "score": <integer from 1 to 5>,
  "reason": "<a brief, concise explanation in English under 50 words>"
}}"""
    return _parse_llm_json(query_astron(system_prompt, user_prompt))

def eval_answer_relevance_with_history(question_used: str, ret_contexts: list, llm_answer: str, history: str = "") -> dict:
    ctx_str = "\n".join([f"Context Block {i+1}: {c}" for i, c in enumerate(ret_contexts)]) if ret_contexts else "No context available."
    system_prompt = "You are an expert conversation relevance evaluator checking conversational flow and prompt adherence."
    user_prompt = f"""Evaluate whether the system's final answer is relevant and conversationally coherent to the given query.

[Dialogue History]:
{history if history else "None"}
[Question Addressed]: {question_used}
[Retrieved Contexts]:
{ctx_str}
[System Actual Response]:
{llm_answer}

Grading Schema (1-5 Scale):
5 - Perfect: Fully relevant and coherent. Directly answers the target intent, maintaining logical continuity.
4 - Good: Fully addresses the question but includes minor wordy or repetitive phrases.
3 - Moderate: Partially relevant. The answer is slightly generic or shifts emphasis away from the core intent.
2 - Poor: Low relevance. Fails to resolve the core intent, generating boilerplate text.
1 - Fail: Irrelevant. Completely ignores the prompt intent.

Output requirements: Return a valid JSON object with exactly two keys:
{{
  "score": <integer from 1 to 5>,
  "reason": "<a brief, concise explanation in English under 50 words>"
}}"""
    return _parse_llm_json(query_astron(system_prompt, user_prompt))