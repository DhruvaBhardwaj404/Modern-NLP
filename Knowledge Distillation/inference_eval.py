from __future__ import annotations

import argparse
import json
import logging
import re
from vllm.lora.request import LoRARequest
import os
from vllm import SamplingParams
from utils import load_vllm_llm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["en", "hindi", "bengali", "kannada", "tamil"]
LANGUAGE_LABELS = {
    "en": "English",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "kannada": "Kannada",
    "tamil": "Tamil",
}
ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*\(([A-J])\)", re.IGNORECASE)
LAST_LINE_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference + eval on test JSONL")
    parser.add_argument("--base_model", required=True,
                        help="Base student model path")
    parser.add_argument("--adapter_path", default="",
                        help="Optional PEFT adapter path")
    parser.add_argument("--test_data", required=True, help="Test JSONL path")
    parser.add_argument("--output_predictions", required=True,
                        help="Predictions JSONL path")
    parser.add_argument("--report_file", required=True,
                        help="Metrics report text file")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def get_cot_ans(output: str):
    output = "<think> " + output
    cot = None
    ans = None
    find_cot = re.search(r'<think>(.*?)</think>', output, re.IGNORECASE | re.DOTALL)
    if find_cot:
        cot = find_cot.group(1).strip()
        find_ans = re.search(r"####\s*ANSWER\s*:\s*\(([A-J])\)", output, re.IGNORECASE)
        if find_ans:
            ans = find_ans.group(1).strip().upper()
    
    return cot,ans

def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    adapter_true = False if args.adapter_path == "" else True
    if "Qwen" in args.base_model:
        name = "qwen"
    else:
        name = "llama"

    if adapter_true:
        llm, tokenizer = load_vllm_llm(
            os.path.join(args.adapter_path,f"{name}_model"),
            tensor_parallel_size=1,
        )
    else:
        llm, tokenizer = load_vllm_llm(
            args.base_model,
            tensor_parallel_size=1,
        )

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    prompts_list = []
    for item in test_data:
        lang = item["language"]
        # system_content = (
        #     f"You are supposed to answer the following MCQ. You must reason in {lang} (Use the respective characters of the language for reasoning):\n"
        #     "1- Provide a deep step-by-step reasoning within <think>...</think> tags.\n"
        #      "2- Provide the final answer in the following format, Correct option should be within the parenthesis: #### ANSWER: (_)"
        # )

        system_content = (
            "You are an expert solver for Multiple Choice Questions. "
            "Follow this exact structure without any deviations or prefix characters:\n"
            "1. Start immediately with your step-by-step reasoning.\n"
            "2. Close your reasoning with the </think> tag.\n"
            "3. On a NEW LINE, provide the final answer using this EXACT template: #### ANSWER: (X)\n"
            "Strictly avoid any leading characters like 'n' or 'Result:' before the #### marker."
            )

        prompts_list.append([
            {"role": "system", "content": system_content},
            {"role": "user", "content": item.get("question", "")},
            {"role": "assistant", "content": "<think>\n"}
        ])

    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False, continue_final_message=True)
               for m in prompts_list]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )


    outputs = llm.generate(prompts, sampling_params)
    generated = [output.outputs[0].text for output in outputs]

    report = []
    lang_wise = {lang: {"correct": 0, "total": 0} for lang in LANGUAGES}

    for item, ans in zip(test_data, generated):
        lang = item["language"]
        true_ans = item["gold_answer"]
        pred_cot, pred_ans = get_cot_ans(ans)

        if lang in lang_wise:
            lang_wise[lang]["total"] += 1
            if pred_ans == true_ans:
                lang_wise[lang]["correct"] += 1

        report.append({
            "language": item.get("language", "en"),
            "question": item.get("question", ""),
            "gold_answer": true_ans,
            "predicted_answer": pred_ans,
            "generation": "<think>\n" + ans
        })


    with open(args.output_predictions, "w") as f:
        for res in report:
            f.write(json.dumps(res) + "\n")

    with open(args.report_file, "w") as f:
        for lang in LANGUAGES:
            stats = lang_wise[lang]
            acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            f.write(f"{LANGUAGE_LABELS[lang]} ACCURACY: {acc:.2f}\n")


if __name__ == "__main__":
    main()
