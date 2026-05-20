from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict
from datasets import Dataset, concatenate_datasets
import random
from data.mmlupro import MMLUPro
from utils import load_vllm_llm, prompt_vllm
import datasets
from vllm import LLM, SamplingParams
import os
os.environ["VLLM_USE_TRITON"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

DEBUG = True

LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.IGNORECASE | re.DOTALL,
)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"({letters[idx]}) {choice}" for idx, choice in enumerate(options)
    )


def sample_datasets(
        samples_per_language: list[int], split, seed
) -> Dataset:
    """Fetch and sample the requested number of rows for each language."""
    if len(samples_per_language) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain 5 comma-separated integers for "
            "english,hindi,bengali,kannada,tamil"
        )
    if any(count < 0 for count in samples_per_language):
        raise ValueError("--num_samples values must be >= 0")
    if sum(samples_per_language) <= 0:
        raise ValueError("--num_samples must request at least one sample")

    print(samples_per_language)
    print(LANGUAGES)
    train_dataset = []
    for lang in LANGUAGES:
        dataset_class = MMLUPro(lang,split="val",)
        dataset =dataset_class.get_dataset()
        subjects = dataset.unique("subject")
        #print(f"unique subs: {subjects} | {len(subjects)}")
        counts = Counter(dataset["subject"])
        #print(counts)
        sub_wise = [dataset.filter(lambda x: x["subject"] == sub) for sub in subjects]
        cur_count = 0
        sub_max =  samples_per_language[LANGUAGES.index(lang)]//len(sub_wise)
        for sub in sub_wise:
            len_sub = len(sub)
            cur_size = min(len_sub,sub_max)
            ind = random.sample(range(len_sub), cur_size)
            train_dataset.extend(sub.select(ind))
            cur_count+=1
            if cur_count == samples_per_language[LANGUAGES.index(lang)]:
                break

        del dataset, counts,subjects,sub_wise


    return Dataset.from_list(train_dataset)


def format_teacher_prompt(question_opt, language: str) -> list:
    """Build a language-aware prompt that enforces reasoning and final answer."""
    system_msg = (
        "You are an expert tutor. Solve the following MCQ. "
        "Your reasoning must be in English, regardless of the question's language.\n"
        "Instructions:\n"
        "1. Start your reasoning immediately after the <think> tag.\n"
        "2. Conclude your reasoning by closing the </think> tag.\n"
        "3. After closing the tag, provide the answer in this exact format: #### ANSWER: (X)"
    )

    user_msg = f"{question_opt}"
    #print(system_msg,user_msg)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "<think>"}
    ]


def parse(outputs: list, true_ans) -> dict[str, str]:
    """Query teacher model once and extract instruction/reasoning/final answer."""
    best_cot = None
    if true_ans is not None:

        best = None
        best_answer = None
        best_len = float('inf')

        for output in outputs:
            output = "<think> " + output
            find_cot = re.search(r'<think>(.*?)</think>', output, re.IGNORECASE | re.DOTALL)
            if find_cot:
                cot = find_cot.group(1).strip()
                find_ans = re.search(r"####\s*ANSWER\s*:\s*\(([A-J])\)", output, re.IGNORECASE | re.DOTALL)
                if find_ans:
                    ans = find_ans.group(1).strip().upper()
                    if ans:
                        best_cot = cot
                        best_answer = ans
                        best = output
                        best_len = len(cot)
            else:
                continue

    if best_cot:
        return {
            "reasoning": best_cot,
            "final_answer": best_answer,
            "raw_generation": best
        }
    else:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query teacher and build train corpus JSONL"
    )
    parser.add_argument(
        "--teacher_model",
        required=True,
        help="Hugging Face path to the teacher model",
    )
    parser.add_argument(
        "--num_samples",
        type=str,
        required=True,
        help=(
            "Comma-separated sample counts for english,hindi,bengali,"
            "kannada,tamil"
        ),
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL path for train corpus",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Target fraction of GPU memory for vLLM; lower if startup fails",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _parse_num_samples(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain exactly 5 comma-separated integers "
            "for english,hindi,bengali,kannada,tamil"
        )

    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "--num_samples must contain only integers"
        ) from exc

    if any(count < 0 for count in counts):
        raise ValueError("--num_samples values must be >= 0")

    return counts


def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"]
    if not isinstance(options, list):
        options = list(options)

    return f"{row['question']}\n\n{_options_to_text(options)}"

def add_column(data):
    data["lang_sub"] = f"{data['language']}_{data['subject']}"
    return data

def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    samples_per_language = _parse_num_samples(args.num_samples)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = sample_datasets(
        samples_per_language=samples_per_language,
        split=args.split,
        seed=args.seed,
    )

    train_dataset = train_dataset.map(add_column)
    train_dataset = train_dataset.cast_column("lang_sub",datasets.ClassLabel(names=list(set(train_dataset["lang_sub"]))))
    dataset = train_dataset.train_test_split(
        test_size=0.2,
        stratify_by_column="lang_sub",
        seed=1
    )
    train_dataset = dataset


    LOGGER.info("Collected %d samples", len(train_dataset))

    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=3000,
    )

    for split in ["train","test"]:
        temp_train = []
        template_prompts = []
        skipped = 0
        lang_wise = defaultdict(int)
        for row in train_dataset[split]:
            question_with_choices = _build_instruction(row)
            prompt = format_teacher_prompt(
                question_with_choices, row["language"])

            full_prompt_str = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True
            )

            token_ids = tokenizer.encode(full_prompt_str)
            if len(token_ids) >= 2300:
                skipped += 1
                lang_wise[row["language"]]+=1
                continue

            temp_train.append({
                "question": question_with_choices,
                "gold_answer": str(row.get("answer", "")).upper()[:1],
                "language": row["language"],
                "subject": row.get("subject"),
                "prompt": prompt,
            })
            template_prompts.append(full_prompt_str)

        print(f"skipped {split}: {skipped}| {lang_wise}")

        sampling_params =  SamplingParams(temperature=0, n=1,max_tokens=375, repetition_penalty=1.1)

        outputs = teacher.generate(template_prompts, sampling_params)
        written = 0
        with open(f"{split}_{str(output_path)}", "w", encoding="utf-8") as fp:
            for ind,output in enumerate(outputs):
                output_gen = [out.text for out in output.outputs]

                parsed = parse(output_gen,temp_train[ind].get("gold_answer",None))
                if parsed is not None:
                    record = {
                        "question": temp_train[ind]["question"],
                        "reasoning": parsed["reasoning"],
                        "final_answer": parsed["final_answer"],
                        "gold_answer": temp_train[ind]["gold_answer"],
                        "language": temp_train[ind]["language"],
                        "subject": temp_train[ind]["subject"],
                        "prompt": temp_train[ind]["prompt"],
                        "teacher_generation": parsed["raw_generation"],
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

            LOGGER.info("Saved %d rows to %s", written, output_path)


if __name__ == "__main__":
    main()
