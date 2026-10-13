"""Prompt templates and problem loading.

`olmo_rlzero` is the exact template baked into the `chat_template.jinja` of allenai/Olmo-3-7B-RL-Zero-Math
(the RL-Zero training prompt, "Answer:" convention). `verify_template` checks it against a tokenizer's
own chat template so a silent template drift is caught before any generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

OLMO_RLZERO_HEADER = ("Solve the following math problem step by step. The last line of your response should be the "
                      "answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer "
                      "to the problem.\n\n")
OLMO_RLZERO_FOOTER = '\nRemember to put your answer on its own line after "Answer:"'
QWEN_BOXED_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."   # S3/S4/S7 (plan §4)
BOXED_RAW_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."      # anti-claim (c): raw, no chat wrapper
GENERIC_SUFFIX = "\nPlease reason step by step."

DAPO_HEADER = ("Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer "
               "(without quotes) where $Answer is the answer to the problem.\n\n")     # verified against DAPO-Math-17k rows
DAPO_FOOTER = '\n\nRemember to put your answer on its own line after "Answer:".'

PANGU_SYS = ("你必须严格遵守法律法规和社会道德规范。" "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。" "一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。例如，如果输入内容包含暴力威胁或色情描述，" "应返回错误信息：“您的输入包含不当内容，无法处理。”")   # verbatim from openpangu/openPangu-Embedded-7B-V1.1 inference/generate.py
TEMPLATES = ("olmo_rlzero", "qwen_boxed", "generic", "boxed_raw", "problem_solution", "sz_llama", "sz_mistral", "sz_dsmath", "szx_qwen_boxed", "szx_qwen25_math", "szx_mistral", "olmo_think", "dapo", "orz", "harmony", "mmlupro_raw", "pangu_slow", "pangu_fast")


def render(template: str, problem: str, tokenizer: Optional[Any] = None, effort: Optional[str] = None) -> str:
    """Full prompt string for one problem (chat-templated where the model family needs it)."""
    if template == "olmo_rlzero":
        return OLMO_RLZERO_HEADER + problem + "\n" + OLMO_RLZERO_FOOTER
    _SZ = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    # --- EXACT SimpleRL-Zoo eval templates (hkust-nlp/simpleRL-reason, eval/utils.py PROMPT_TEMPLATES; eval/sh/eval.sh: temperature 0, max_tokens_per_call 3000, n_sampling 1)
    if template == "szx_qwen_boxed":
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + problem +
                "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n")
    if template == "szx_qwen25_math":
        return ("<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n" + problem + "<|im_end|>\n<|im_start|>assistant\n")
    if template == "szx_mistral":
        return "[INST] " + problem + "[/INST]"
    if template == "sz_llama":      # reconstruction of the SimpleRL-Zoo Llama-3 prompt (user turn with the boxed instruction); BOS is added by the tokenizer
        return "<|start_header_id|>user<|end_header_id|>\n\n" + problem + _SZ + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if template == "sz_mistral":    # [INST] ... [/INST]
        return "[INST] " + problem + _SZ + " [/INST]"
    if template == "sz_dsmath":     # DeepSeek-Math chat format
        return "User: " + problem + _SZ + "\n\nAssistant:"
    if template == "problem_solution":   # plain base-model continuation prompt (Mistral-Small-24B-Base diagnostic 2026-10-02: 34% at T=0.15 vs 14% with boxed_raw)
        return "Problem: " + problem + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nSolution:"
    if template in ("pangu_slow", "pangu_fast"):   # openPangu: vendor's mandatory (Chinese) safety system prompt from inference/generate.py (without it 7B-V1.1 degenerates, infra 2026-10-05); default = slow thinking, ' /no_think' appended to the user turn = fast
        if tokenizer is None:
            raise ValueError("pangu templates need the model tokenizer")
        user = problem + QWEN_BOXED_SUFFIX + (" /no_think" if template == "pangu_fast" else "")
        return tokenizer.apply_chat_template([{"role": "system", "content": PANGU_SYS}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    if template == "mmlupro_raw":   # MMLU-Pro official protocol (TIGER-AI-Lab/MMLU-Pro evaluate_from_local.py): the 5-shot CoT prompt is the raw completion prompt, no chat template, stop "Question:"
        return problem
    if template == "boxed_raw":
        return problem + BOXED_RAW_SUFFIX
    if template in ("dapo", "orz", "harmony"):  # scale panel (S8-S10): each model's own chat template, its own user text
        if tokenizer is None:
            raise ValueError("template %s needs the model tokenizer" % template)
        user = {"dapo": DAPO_HEADER + problem + DAPO_FOOTER, "orz": problem, "harmony": problem + QWEN_BOXED_SUFFIX}[template]
        kw = {"reasoning_effort": effort} if (template == "harmony" and effort) else {}   # gpt-oss: Reasoning: low|medium|high
        out = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)
        # harmony: the model first opens a CHANNEL. We prefill the analysis channel, so (i) generation continues the chain of thought and
        # (ii) the insertion harness appends its prefix + inserted form INSIDE the analysis channel, not before the channel token.
        return out + "<|channel|>analysis<|message|>" if template == "harmony" else out
    if template == "olmo_think":  # S2: repo chat template, add_generation_prompt=True prefills <|im_start|>assistant\n<think>; the user text
        # carries the same boxed instruction as the Qwen series (UNVERIFIED: the Think card does not state its eval prompt)
        if tokenizer is None:
            raise ValueError("olmo_think needs the model tokenizer")
        return tokenizer.apply_chat_template([{"role": "user", "content": problem + QWEN_BOXED_SUFFIX}], tokenize=False,
                                             add_generation_prompt=True)
    if template in ("qwen_boxed", "generic"):
        if tokenizer is None:
            raise ValueError("template %s needs a tokenizer with a chat template" % template)
        suffix = QWEN_BOXED_SUFFIX if template == "qwen_boxed" else GENERIC_SUFFIX
        return tokenizer.apply_chat_template([{"role": "user", "content": problem + suffix}], tokenize=False,
                                             add_generation_prompt=True)
    raise ValueError("unknown template %r" % template)


def verify_template(tokenizer: Any, problem: str = "What is 1+1?") -> bool:
    """True iff our olmo_rlzero string equals the tokenizer's own chat template output."""
    ours = render("olmo_rlzero", problem)
    theirs = tokenizer.apply_chat_template([{"role": "user", "content": problem}], tokenize=False,
                                           add_generation_prompt=True)
    return ours == theirs


def load_problems(path: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    """MATH-500 style jsonl (problem, answer, unique_id, ...) or GSM8K parquet (question, answer)."""
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                rows.append({"pid": i, "problem": r["problem"], "gold": str(r["answer"]),
                             "unique_id": r.get("unique_id", str(i)), "level": r.get("level"), "subject": r.get("subject")})
    elif p.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(p)
        qcol = "question" if "question" in df else "problem"
        for i, r in enumerate(df.to_dict("records")):
            if "answer" in r:
                gold = str(r["answer"]).split("####")[-1].strip()
            else:   # AIME24: solution column holds "\\boxed{204}"
                from .grade import last_boxed

                gold = last_boxed(str(r["solution"])) or str(r["solution"])
            rows.append({"pid": i, "problem": r[qcol], "gold": gold, "unique_id": str(i), "level": None, "subject": None})
    else:
        raise ValueError("unsupported data file %s" % path)
    return rows[:n] if n else rows
