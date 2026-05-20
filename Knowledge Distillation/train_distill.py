from __future__ import annotations
import argparse
import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, concatenate_datasets
import torch.nn.functional as F
import gc
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)
import re

import warnings

warnings.filterwarnings("ignore")

TRAIN_SFT = True
TRAIN_on_policy = True

MAX_SUB = 4


def merge_adapters(output_dir, model_name):
    if "Qwen" in model_name:
        name = "qwen"
    else:
        name = "llama"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    adapters = [f"SFT_{name}_final", f"on_policy_{name}_final"]  # ,"GRPO_{name}_final"]

    for i, aname in enumerate(adapters):
        model = PeftModel.from_pretrained(model, os.path.join(output_dir, aname))
        model = model.merge_and_unload()

    model.save_pretrained(os.path.join(output_dir, f"{name}_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, f"{name}_model"))


class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        tcorrect = [f.pop("tcorrect") for f in features]
        correct = [f.pop("correct") for f in features]
        input_ids_list = [torch.tensor(f["input_ids"]) for f in features]
        completion_mask_list = [torch.tensor(f["completion_mask"]) for f in features]
        max_len = max(x.shape[0] for x in input_ids_list)
        input_ids = torch.stack(
            [F.pad(x, (max_len - x.shape[0], 0), value=self.tokenizer.pad_token_id) for x in input_ids_list])
        completion_mask = torch.stack([F.pad(x, (max_len - x.shape[0], 0), value=0) for x in completion_mask_list])
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[completion_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tcorrect": tcorrect,
            "correct": correct,
        }


@dataclass
class customOnPolicyConfig(SFTConfig):
    lambda_: float = field(default=1.0)
    beta: float = field(default=0.3)
    gamma: float = field(default=0.2)
    temp: float = field(default=3.0)
    uld_loss: bool = field(default=True)
    uld_top_k: int = field(default=50)
    kl_top_k: int = field(default=200)


class customOnPolicyTrainer(SFTTrainer):
    def __init__(self, tmodel, ttok, same_family, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.tmodel: AutoModelForCausalLM = tmodel
        self.ttok: AutoTokenizer = ttok
        self.stok = self.processing_class
        self.same_fam = same_family
        self.stok.padding_side = "left"
        if not self.same_fam:
            self.create_vocab_map_custom()
            self.cross_map = self.cross_map.to(self.model.device)
            self.map_tensor = self.map_tensor.to(self.model.device)
            self.map_len = self.map_len.to(self.model.device)
            self.map_valid = self.map_valid.to(self.model.device)

        self.tmodel.eval()
        for p in self.tmodel.parameters():
            p.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        tcorrect = inputs.pop("tcorrect", None)
        # correct = inputs.pop("correct", None)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        soutput = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss_sft = soutput.loss
        if not return_outputs:
            del soutput
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            model.config.use_cache = True
            self.stok.padding_side = "left"
            with torch.amp.autocast("cuda", dtype=torch.float16):
                gen_id = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=375,
                    do_sample=False,
                    pad_token_id=self.stok.pad_token_id,
                    eos_token_id=self.stok.eos_token_id,
                    repetition_penalty=1.1,
                )
            model.config.use_cache = False
            gen_tok = gen_id[:, prompt_len:]
            decoded = self.stok.batch_decode(gen_tok, skip_special_tokens=True)

            tcorrect_tensor = torch.tensor(tcorrect, device=model.device, dtype=torch.float32)
            sgen_mask = (gen_id[:, prompt_len:] != self.stok.pad_token_id).float()
            distill_mask = sgen_mask * tcorrect_tensor.view(-1, 1)

            if self.same_fam:
                tlogits = self.tmodel(gen_id, attention_mask=(gen_id != self.stok.pad_token_id)).logits
                tlogits = tlogits[:, prompt_len - 1:-1]
            else:
                decoded = [t if t.strip() else " " for t in decoded]
                teacher_inputs = self.ttok(decoded, return_tensors="pt", padding=True, truncation=True).to(model.device)
                tlogits_full = self.tmodel(**teacher_inputs).logits
                align_tlogits, align_tprobs, align_mask = self.align(
                    decoded, gen_tok, tlogits_full, teacher_inputs["input_ids"].long()
                )
                distill_mask = distill_mask * align_mask.float()
        del gen_tok
        slogits = model(gen_id, attention_mask=(gen_id != self.stok.pad_token_id)).logits
        slogits_temp = slogits[:, prompt_len - 1:-1].float()
        del gen_id, slogits

        min_len = min(slogits_temp.shape[1], distill_mask.shape[1])
        if self.same_fam:
            min_len = min(min_len, tlogits.shape[1])
            tlogits_temp = tlogits[:, :min_len]
        else:
            min_len = min(min_len, align_tlogits.shape[1])
            tlogits_temp = align_tlogits[:, :min_len]

        slogits_temp = slogits_temp[:, :min_len]
        distill_mask = distill_mask[:, :min_len]

        valid_ids = distill_mask > 0.0
        slogits_flat = slogits_temp[valid_ids]
        tlogits_flat = tlogits_temp[valid_ids]
        mask_flat = distill_mask[valid_ids]

        if self.same_fam:
            loss_kl, loss_uld = self.get_kl_uld_loss_custom(slogits_flat, tlogits_flat, mask_flat)
        else:
            tprobs_temp = align_tprobs[:, :min_len]
            tprobs_flat = tprobs_temp[valid_ids]
            loss_kl, loss_uld = self.get_kl_uld_cross_custom(slogits_flat, tlogits_flat, tprobs_flat, mask_flat)

        loss_distill = (self.args.beta * loss_kl) + (self.args.gamma * loss_uld)
        total_loss = (self.args.lambda_ * loss_sft) + loss_distill
        return (total_loss, soutput) if return_outputs else total_loss

    def get_kl_uld_loss_custom(self, slogits, tlogits, mask):
        if slogits.numel() == 0:
            return torch.tensor(0.0, device=slogits.device), torch.tensor(0.0, device=slogits.device)

        min_vocab = min(slogits.shape[-1], tlogits.shape[-1])
        slogits = slogits[:, :min_vocab]
        tlogits = tlogits[:, :min_vocab]

        with torch.no_grad():
            prob_teacher = F.softmax(tlogits / self.args.temp, dim=-1)
            max_k = max(self.args.kl_top_k, self.args.uld_top_k if self.args.uld_loss else 0)
            top_tprob, top_tid = prob_teacher.topk(max_k, dim=-1)

        slogits_scaled = slogits / self.args.temp
        s_lse = torch.logsumexp(slogits_scaled, dim=-1, keepdim=True)

        kl_tprob = top_tprob[:, :self.args.kl_top_k]
        kl_tid = top_tid[:, :self.args.kl_top_k]
        kl_tprob_norm = kl_tprob / kl_tprob.sum(dim=-1, keepdim=True)
        slogits_gathered = slogits_scaled.gather(-1, kl_tid)
        stop_prob = slogits_gathered - s_lse
        kl_per_token = -(kl_tprob_norm * stop_prob).sum(dim=-1)
        loss_kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1e-8)
        loss_kl = loss_kl * (self.args.temp ** 2)

        loss_uld = torch.tensor(0.0, device=slogits.device)
        if self.args.uld_loss:
            uld_tprob = top_tprob[:, :self.args.uld_top_k]
            uld_tid = top_tid[:, :self.args.uld_top_k]

            slogits_uld_gathered = slogits_scaled.gather(-1, uld_tid)
            top_s_logprob = slogits_uld_gathered - s_lse

            loss_per_token = -(uld_tprob * top_s_logprob).sum(dim=-1)
            teacher_conf = prob_teacher.max(dim=-1).values
            loss_per_token = loss_per_token * teacher_conf
            loss_uld = (loss_per_token * mask).sum() / mask.sum().clamp(min=1e-8)
            loss_uld = loss_uld * (self.args.temp ** 2)

        return loss_kl, loss_uld

    def get_kl_uld_cross_custom(self, slogits, tlogits, tprobs, mask):
        if slogits.numel() == 0:
            return torch.tensor(0.0, device=slogits.device), torch.tensor(0.0, device=slogits.device)

        merge_weight = tprobs.exp().clamp(max=1.0)
        weighted_mask = mask.float() * merge_weight

        with torch.no_grad():
            tprob = F.softmax(tlogits.float() / self.args.temp, dim=-1)
            max_k = max(self.args.kl_top_k, self.args.uld_top_k if self.args.uld_loss else 0)
            top_tprob, top_tid = tprob.topk(max_k, dim=-1)

        slogits_scaled = slogits / self.args.temp
        s_lse = torch.logsumexp(slogits_scaled, dim=-1, keepdim=True)
        top_tid_clamped = top_tid.clamp(0, self.cross_map.shape[0] - 1)
        matched_sid = self.cross_map[top_tid_clamped]
        true_match = matched_sid >= 0
        matched_sid_clamped = matched_sid.clamp(min=0)

        kl_tprob = top_tprob[:, :self.args.kl_top_k]
        kl_match_sid = matched_sid_clamped[:, :self.args.kl_top_k]
        kl_true_match = true_match[:, :self.args.kl_top_k]
        slogits_gathered = slogits_scaled.gather(-1, kl_match_sid)
        stop_prob_kl = slogits_gathered - s_lse
        matched_tprob = kl_tprob * kl_true_match.float()
        matched_tprob_norm = matched_tprob / matched_tprob.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        kl_matched = -(matched_tprob_norm * stop_prob_kl).sum(dim=-1)
        has_matched = kl_true_match.any(dim=-1).float()
        loss_kl = (kl_matched * weighted_mask * has_matched).sum() / weighted_mask.sum().clamp(min=1e-8)
        loss_kl = loss_kl * (self.args.temp ** 2)

        loss_uld = torch.tensor(0.0, device=slogits.device)
        if self.args.uld_loss:
            uld_tprob = top_tprob[:, :self.args.uld_top_k]
            uld_true_match = true_match[:, :self.args.uld_top_k]
            unmatched_tprob = uld_tprob * (~uld_true_match).float()
            top_s_logits, _ = slogits_scaled.topk(self.args.uld_top_k, dim=-1)
            top_s_logprob = top_s_logits - s_lse

            k = min(self.args.uld_top_k, unmatched_tprob.shape[-1], top_s_logprob.shape[-1])
            unmatched_tprob_k = unmatched_tprob[:, :k]
            top_s_logprob_k = top_s_logprob[:, :k]

            uld_per_token = -(unmatched_tprob_k * top_s_logprob_k).sum(dim=-1)
            has_unmatched = (~uld_true_match).any(dim=-1).float()

            loss_uld = (uld_per_token * weighted_mask * has_unmatched).sum() / weighted_mask.sum().clamp(min=1e-8)
            loss_uld = loss_uld * (self.args.temp ** 2)

        return loss_kl, loss_uld

    def create_vocab_map_custom(self):
        tvocab_size = self.ttok.vocab_size
        svocab_size = self.stok.vocab_size

        str_id_stud = {}
        for sid in range(svocab_size):
            tok_str = self.stok.convert_ids_to_tokens(sid)
            if tok_str is not None:
                str_id_stud[tok_str] = sid

        self.cross_map = torch.full((tvocab_size,), -1, dtype=torch.long)
        for tid in range(tvocab_size):
            tok_str = self.ttok.convert_ids_to_tokens(tid)
            if tok_str is not None and tok_str in str_id_stud:
                self.cross_map[tid] = str_id_stud[tok_str]

        all_texts = self.ttok.batch_decode(list(range(tvocab_size)), skip_special_tokens=True)
        valid_tids, valid_texts = [], []
        for tid, tok_text in enumerate(all_texts):
            if tok_text.strip():
                valid_tids.append(tid)
                valid_texts.append(tok_text)

        encoded = self.stok(valid_texts, add_special_tokens=False, return_attention_mask=False)

        self.map_tensor = torch.full((tvocab_size, MAX_SUB), self.stok.pad_token_id, dtype=torch.long)
        self.map_len = torch.ones(tvocab_size, dtype=torch.long)
        self.map_valid = torch.zeros(tvocab_size, dtype=torch.bool)

        for tid, s_ids_list in zip(valid_tids, encoded["input_ids"]):
            if s_ids_list and tid < tvocab_size:
                n = min(len(s_ids_list), MAX_SUB)
                self.map_tensor[tid, :n] = torch.tensor(s_ids_list[:n], dtype=torch.long)
                self.map_tensor[tid, n:] = self.stok.pad_token_id
                self.map_len[tid] = n
                self.map_valid[tid] = True

    def align(self, decoded_texts, gen_tok, tlogits_full, t_ids_batch):
        device = gen_tok.device
        B, S = gen_tok.shape
        t_vocab = tlogits_full.shape[-1]

        senc = self.stok(decoded_texts, return_offsets_mapping=True, add_special_tokens=False, padding='max_length',
                         max_length=S, truncation=True)
        tenc = self.ttok(decoded_texts, return_offsets_mapping=True, add_special_tokens=False,
                         padding='max_length', max_length=tlogits_full.shape[1], truncation=True)

        soff = torch.tensor(senc["offset_mapping"], device=device)
        toff = torch.tensor(tenc["offset_mapping"], device=device)

        s_start, s_end = soff[..., 0].unsqueeze(2), soff[..., 1].unsqueeze(2)
        t_start, t_end = toff[..., 0].unsqueeze(1), toff[..., 1].unsqueeze(1)
        overlap = (t_end > s_start) & (t_start < s_end) & ((s_end - s_start) > 0)
        tids = torch.arange(toff.size(1), device=device).view(1, 1, -1)
        mask_tid = torch.where(overlap, tids, torch.tensor(-1, device=device))
        last_ti = mask_tid.max(dim=2).values
        valid_mask = (last_ti >= 0)
        gather_idx = last_ti.unsqueeze(-1).expand(-1, -1, t_vocab)

        ids = last_ti.clamp(min=0).unsqueeze(-1).expand(-1, -1, t_vocab)
        align_tlogits = torch.gather(tlogits_full.float(), 1, ids)
        align_tlogits = align_tlogits * valid_mask.unsqueeze(-1)
        tprob = F.log_softmax(tlogits_full.float(), dim=-1)
        tid_next = t_ids_batch[:, 1:].contiguous()
        tprob_step = torch.gather(tprob[:, :-1, :], 2, tid_next.unsqueeze(-1)).squeeze(-1)
        tprob_step = F.pad(tprob_step, (0, 1), value=0.0)

        last_overlap = overlap & (tids < last_ti.unsqueeze(2))
        align_tprobs = (tprob_step.unsqueeze(1) * last_overlap).sum(dim=2)
        del overlap, s_start, s_end, t_start, t_end
        # torch.cuda.empty_cache()

        return align_tlogits.half(), align_tprobs, valid_mask


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Starter distillation training loop")
    parser.add_argument("--student_model", required=True,
                        help="Base student model path")
    parser.add_argument("--teacher_model", required=False,
                        help=" model path")
    parser.add_argument("--train_data", default="data/train.jsonl",
                        help="Path to train JSONL with prompt_ and _generation")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save trained weights")
    # parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    # parser.add_argument("--epochs", type=int, default=5, help="Epoch count")
    # parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    # parser.add_argument("--max_length", type=int,
    #                     default=2048, help="Max sequence length")
    parser.add_argument(
        "--mask_prompt__tokens",
        action="store_true",
        help="Mask prompt_ tokens so loss is only on reasoning + final answer",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def format_prompt(items, tok):
    prompts = []
    ans = []
    tcorrect = []
    correct_ans = items["gold_answer"]
    lang = items["language"]
    for i in range(len(items["question"])):
        system_content = (
            "You are an expert solver for Multiple Choice Questions. "
            "Follow this exact structure without any deviations or prefix characters:\n"
            "1. Start immediately with your step-by-step reasoning.\n"
            "2. Close your reasoning with the </think> tag.\n"
            "3. On a NEW LINE, provide the final answer using this EXACT template: #### ANSWER: (X)\n"
            "Strictly avoid any leading characters like 'n' or 'Result:' before the #### marker."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": items["question"][i]},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        cur_ans = f"<think>\n{items['reasoning'][i]}\n</think>\n#### ANSWER: ({items['final_answer'][i]})"
        correct = (items["gold_answer"][i] == items["final_answer"][i])
        tcorrect.append(correct)
        prompts.append(prompt)
        ans.append(cur_ans)

    return prompts, ans, tcorrect, correct_ans, lang


def distill_Qwen(args):
    t1 = time.time()
    teacher = args.teacher_model
    student = args.student_model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    student_model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float16, device_map={"": 0})
    student_model.config.use_cache = False
    lora_config = LoraConfig(r=64, lora_alpha=128,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                             "down_proj"],
                             lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
                             )

    dataset_train = load_dataset("json", data_files=os.path.join(args.train_data, "train_data.jsonl"))["train"]
    # dataset_valid = load_dataset("json", data_files=f"test_{args.train_data}")["train"]

    train_prompts, train_ans, train_tcorrect, train_correct, train_lang = format_prompt(dataset_train.to_dict(),
                                                                                        tokenizer)
    # valid_prompts, valid_ans, valid_tcorrect, valid_correct, valid_lang = format_prompt(dataset_valid.to_dict(),tokenizer)

    train_dataset = Dataset.from_dict(
        {"prompt": train_prompts, "completion": train_ans, "tcorrect": train_tcorrect, "correct": train_correct,
         "language": train_lang})
    # valid_dataset = Dataset.from_dict({"prompt": valid_prompts, "completion": valid_ans, "tcorrect":valid_tcorrect,"correct":valid_correct, "language":valid_lang})

    sft_train_dataset = train_dataset.filter(lambda example: example["tcorrect"] is True)
    # sft_valid_dataset = valid_dataset.filter(lambda example: example["tcorrect"] is True)

    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "SFT_qwen"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        fp16=True,
        # max_steps=2,
        gradient_checkpointing=True,
        eval_strategy="no",
        save_strategy="no",
        # max_seq_length=1400,
        optim="adamw_8bit",
        logging_steps=10,
        report_to="none",
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        completion_only_loss=True,
    )

    trainerSFT = SFTTrainer(
        model=student_model,
        train_dataset=sft_train_dataset,
        # eval_dataset=sft_valid_dataset,
        peft_config=lora_config,
        args=sft_config,
        processing_class=tokenizer,
    )
    del sft_train_dataset  # , sft_valid_dataset

    if TRAIN_SFT:
        trainerSFT.train()
        trainerSFT.save_model(os.path.join(output_dir, "SFT_qwen_final"))
        tokenizer.save_pretrained(os.path.join(output_dir, "SFT_qwen_final"))

    del student_model
    student_model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float16, device_map={"": 0})
    student_model_temp = PeftModel.from_pretrained(student_model, os.path.join(output_dir, "SFT_qwen_final"))
    student_model_lora = student_model_temp.merge_and_unload()
    student_model_lora.generation_config.padding_side = "left"
    del tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "SFT_qwen_final"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    student_model_lora.config.use_cache = False
    del trainerSFT
    del student_model
    del student_model_temp
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tmodel = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype=torch.float16, device_map={"": 0},
                                                  quantization_config=quant_config)
    ttok = AutoTokenizer.from_pretrained(teacher)
    tmodel.config.use_cache = False
    tmodel.eval()

    lora_config_on_policy = LoraConfig(r=32, lora_alpha=64,
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                       "down_proj"],
                                       lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
                                       )

    on_policy_config = customOnPolicyConfig(
        output_dir=os.path.join(output_dir, "on_policy_qwen"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        num_train_epochs=1,
        max_grad_norm=1.0,
        fp16=True,
        # max_steps=2,
        optim="adamw_8bit",
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        eval_strategy="no",
        save_strategy="no",
        warmup_ratio=0.1,
        logging_steps=10,
        # max_seq_length=1200,
        lambda_=1.0,
        beta=2.5,
        gamma=0.0,
        temp=1.5,
        uld_loss=False,
        uld_top_k=50,
        kl_top_k=128,
        completion_only_loss=False,
        remove_unused_columns=False,
    )

    lang_wise_data = []
    langs = ["en", "hindi", "bengali", "kannada", "tamil"]
    n = 800
    per_lang = n // len(langs)

    for lang in langs:
        cur_lang_data = train_dataset.filter(lambda x: x['language'] == lang)
        tcorrect = cur_lang_data.filter(lambda x: x['tcorrect'] == True)

        if len(tcorrect) >= per_lang:
            segment = tcorrect.shuffle(seed=2).select(range(per_lang))
        else:
            segment = cur_lang_data.shuffle(seed=2).select(range(min(len(cur_lang_data), per_lang)))

        lang_wise_data.append(segment)

    train_dataset_on_policy = concatenate_datasets(lang_wise_data).shuffle(seed=2)
    # train_dataset_on_policy = train_dataset.shuffle(seed=2).select(range(n))

    traineron_policy = customOnPolicyTrainer(
        model=student_model_lora,
        tmodel=tmodel,
        ttok=ttok,
        args=on_policy_config,
        same_family=True,
        processing_class=tokenizer,
        train_dataset=train_dataset_on_policy,
        # eval_dataset=valid_dataset,
        peft_config=lora_config_on_policy,
        data_collator=CustomCollator(tokenizer),
    )
    if TRAIN_on_policy:
        traineron_policy.train()
        traineron_policy.save_model(os.path.join(output_dir, "on_policy_qwen_final"))
        tokenizer.save_pretrained(os.path.join(output_dir, "on_policy_qwen_final"))

    del traineron_policy, tokenizer
    del train_dataset_on_policy
    del tmodel, ttok
    del student_model_lora

    merge_adapters(output_dir, student)
    tt = (time.time() - t1) / 60
    print(f"Time taken {tt} mins", )


def distill_Llama(args):
    t1 = time.time()
    teacher = args.teacher_model
    student = args.student_model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    student_model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.float16, device_map={"": 0})
    student_model.config.use_cache = False
    lora_config = LoraConfig(r=64, lora_alpha=128,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                             "down_proj"],
                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
                             )

    dataset_train = load_dataset("json", data_files=os.path.join(args.train_data, "train_data.jsonl"))["train"]
    # dataset_valid = load_dataset("json", data_files=f"test_{args.train_data}")["train"]

    train_prompts, train_ans, train_tcorrect, train_correct, train_lang = format_prompt(dataset_train.to_dict(),
                                                                                        tokenizer)
    # valid_prompts, valid_ans, valid_tcorrect, valid_correct, valid_lang = format_prompt(dataset_valid.to_dict(), tokenizer)

    train_dataset = Dataset.from_dict(
        {"prompt": train_prompts, "completion": train_ans, "tcorrect": train_tcorrect, "correct": train_correct,
         "language": train_lang})
    # valid_dataset = Dataset.from_dict(
    #   {"prompt": valid_prompts, "completion": valid_ans, "tcorrect": valid_tcorrect, "correct": valid_correct, "language":valid_lang})

    sft_train_dataset = train_dataset.filter(lambda example: example["tcorrect"] is True)
    # sft_valid_dataset = valid_dataset.filter(lambda example: example["tcorrect"] is True)

    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "SFT_llama"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=True,
        # max_steps=1,
        gradient_checkpointing=True,
        eval_strategy="no",
        save_strategy="no",
        # max_seq_length=3000,
        logging_steps=10,
        report_to="none",
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        completion_only_loss=True,
    )

    trainerSFT = SFTTrainer(
        model=student_model,
        train_dataset=sft_train_dataset,
        # eval_dataset=sft_valid_dataset,
        peft_config=lora_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    if TRAIN_SFT:
        trainerSFT.train()
        trainerSFT.save_model(os.path.join(output_dir, "SFT_llama_final"))
        tokenizer.save_pretrained(os.path.join(output_dir, "SFT_llama_final"))

    del sft_train_dataset  # , sft_valid_dataset
    del trainerSFT
    gc.collect()
    torch.cuda.empty_cache()

    student_model = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=torch.float16,
                                                         device_map={"": 0})
    student_model_temp = PeftModel.from_pretrained(student_model, os.path.join(output_dir, "SFT_llama_final"))
    student_model_lora = student_model_temp.merge_and_unload()
    student_model_lora.generation_config.padding_side = "left"
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "SFT_llama_final"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    student_model_lora.config.use_cache = False
    del student_model
    del student_model_temp
    gc.collect()
    torch.cuda.empty_cache()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tmodel = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype=torch.float16, device_map={"": 0},
                                                  quantization_config=quant_config)
    ttok = AutoTokenizer.from_pretrained(teacher)
    tmodel.config.use_cache = False
    tmodel.eval()

    lora_config_on_policy = LoraConfig(r=32, lora_alpha=64,
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                       "down_proj"],
                                       lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
                                       )

    lang_wise_data = []
    langs = ["en", "hindi", "bengali", "kannada", "tamil"]
    n = 1500
    per_lang = n // len(langs)

    for lang in langs:
        cur_lang_data = train_dataset.filter(lambda x: x['language'] == lang)
        tcorrect = cur_lang_data.filter(lambda x: x['tcorrect'] == True)

        if len(tcorrect) >= per_lang:
            segment = tcorrect.shuffle(seed=2).select(range(per_lang))
        else:
            segment = cur_lang_data.shuffle(seed=2).select(range(min(len(cur_lang_data), per_lang)))

        lang_wise_data.append(segment)

    train_dataset_on_policy = concatenate_datasets(lang_wise_data).shuffle(seed=2)

    on_policy_config = customOnPolicyConfig(
        lr_scheduler_type="cosine",
        output_dir=os.path.join(output_dir, "on_policy_llama"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_grad_norm=1.0,
        gradient_accumulation_steps=4,
        optim="adamw_8bit",
        learning_rate=1e-5,
        num_train_epochs=1,
        warmup_ratio=0.1,
        fp16=True,
        # max_steps=1,
        gradient_checkpointing=True,
        # max_seq_length=1400,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        lambda_=1.0,
        beta=0.8,
        gamma=0.6,
        temp=2.5,
        uld_loss=True,
        uld_top_k=32,
        kl_top_k=64,
        completion_only_loss=False,
        remove_unused_columns=False,
    )

    traineron_policy = customOnPolicyTrainer(
        model=student_model_lora,
        tmodel=tmodel,
        ttok=ttok,
        args=on_policy_config,
        same_family=False,
        processing_class=tokenizer,
        train_dataset=train_dataset_on_policy,
        # eval_dataset=valid_dataset,
        peft_config=lora_config_on_policy,
        data_collator=CustomCollator(tokenizer)
    )

    if TRAIN_on_policy:
        traineron_policy.train()
        traineron_policy.save_model(os.path.join(output_dir, "on_policy_llama_final"))
        tokenizer.save_pretrained(os.path.join(output_dir, "on_policy_llama_final"))

    del traineron_policy
    del train_dataset_on_policy
    del tmodel, ttok
    del student_model_lora

    merge_adapters(output_dir, student)
    tt = (time.time() - t1) / 60
    print(f"Time taken {tt} mins", )


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    if "Qwen" in args.student_model:
        distill_Qwen(args)
    else:
        distill_Llama(args)


if __name__ == "__main__":
    main()
