import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.nn.utils.rnn import pad_sequence
import unicodedata
import json
import random
from collections import Counter
import difflib
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import warnings
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import math
warnings.filterwarnings("ignore")


DEBUG = False
DEBUG_FINE_TUNE = False
IGNORE_INDEX = -100
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 4
INFER_BATCH_SIZE = 8
PRETRAIN_BATCH_SIZE = 4
PAD_TOKEN_ID = None
ACCUMALATE_BATCHES = 16
WARMUP_RATIO = 0.05
MAX_LEN = 1024
LANG_LIST = ["en","hi","kn","or","tcy"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normalise(text):
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text

def normalise_accent(text):
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text.lower()


def get_marked_sent(sent, entity, marker):
    if entity in sent:
        return sent.replace(entity, f"<{marker}>{entity}</{marker}>", 1)

    ind = normalise_accent(sent).find(normalise_accent(entity))
    if ind != -1:
        org_sent = sent[ind:ind + len(entity)]
        return sent[:ind] + f"<{marker}>{org_sent}</{marker}>" + sent[ind + len(entity):]

    return None

def create_pretrain_dataset(data, tokenizer, num_tuples=None, lang_id=1):
    dataset = []
    train = data["train"]

    n = min(num_tuples, len(train)) if num_tuples else len(train)
    train = train.select(range(n))

    for b in tqdm(train):
        text = normalise(b["text"].strip())
        if not text:
            continue

        text_tok = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
            padding=False,
        )
        input_ids = text_tok["input_ids"].squeeze(0)
        attn_mask = text_tok["attention_mask"].squeeze(0)
        labels = input_ids.clone()   

        dataset.append({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "lang_id":lang_id
        })
        
    return CustomDataset(dataset)


def custom_collate2(batch):
    global PAD_TOKEN_ID, IGNORE_INDEX

    input_ids = [b["input_ids"] for b in batch]
    attn_masks = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    lang_id = [b["lang_id"] for b in batch]
    input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attn_mask_pad = pad_sequence(attn_masks,     batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels,    batch_first=True, padding_value=IGNORE_INDEX)
    lang_id_tensor = torch.tensor(lang_id, dtype = torch.float32)
    return {
        "input_ids": input_ids_pad,
        "attention_mask": attn_mask_pad,
        "labels": labels_pad,
        "lang_id":lang_id_tensor
    }


class LoraModel(torch.nn.Module):
    def __init__(self, tok_len, max_gen_tokens):
        super().__init__()

        qwen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.float16)
        qwen_model.resize_token_embeddings(tok_len)

        qwen_model.gradient_checkpointing_enable()
        qwen_model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.max_gen_tokens = max_gen_tokens
        self.lora_model= get_peft_model(qwen_model, lora_config)


    def forward(self, input_ids, attn_mask,labels=None):
        return self.lora_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels
        )

    def generate(self, input_ids, attn_mask):
        return self.lora_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_gen_tokens,
            do_sample=False,
            pad_token_id=PAD_TOKEN_ID,
        )


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id]


def custom_collate1(batch):
    input_ids = [b["input_ids"] for b in batch]
    attn_masks = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    label_text = [b["label_str"] for b in batch]
    input_lens = [b["prompt_len"] for b in batch]
    label_ids = torch.tensor([b["label_id"] for b in batch])

    input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attn_masks_pad = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    label_id = torch.tensor(label_ids, dtype=torch.long)

    return {
        "input_ids": input_ids_pad,
        "attention_mask": attn_masks_pad,
        "labels": labels_pad,
        "label_text": label_text,
        "input_lens": input_lens,
        "label_id": label_id
    }

def create_dataset(data_path, tokenizer, label2id, english=False, val=False, label_map=None):

    label_count = defaultdict(int)
    label_tuples = defaultdict(list)

    not_count = 0
    with open(data_path, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue

            b = json.loads(line)
            sentence: str = normalise(b["sentText"].strip())

            for rm in b["relationMentions"]:
                e1 = normalise(rm["em1Text"])
                e2 = normalise(rm["em2Text"])
                try:
                    if label_map is None:
                        label_str = normalise(rm["label"])
                    else:
                        label_str = normalise(label_map[rm["label"]])
                    label_count[label_str] += 1
                except:
                    not_count += 1
                    continue
                try:
                    label_id = label2id[label_str]
                except:
                    not_count+=1
                    print("*"*10)
                    print(label_str)
                    print("*" * 10)
                    continue

                if len(e2) > len(e1):
                    sentence_cur = get_marked_sent(sentence, e2, "E2")
                    if sentence_cur:
                        sentence_cur = get_marked_sent(sentence_cur, e1, "E1")
                else:
                    sentence_cur = get_marked_sent(sentence, e1, "E1")
                    if sentence_cur:
                        sentence_cur = get_marked_sent(sentence_cur, e2, "E2")

                if sentence_cur is None:
                    not_count += 1
                    continue

                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Output the relationship between the entities > format: Entity_Type/Entity_Type/Relationship_Label"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"sentence: {sentence_cur}\nrelation:",
                    },
                ]

                prompt = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                full_prompt = prompt + label_str + tokenizer.eos_token

                full_inp_ids = tokenizer(
                    full_prompt,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                    padding=False,
                )
                input_ids = full_inp_ids["input_ids"].squeeze(0)
                attn_mask = full_inp_ids["attention_mask"].squeeze(0)

                prompt_enc = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                    padding=False,
                )
                prompt_len = prompt_enc["input_ids"].shape[1]

                labels = input_ids.clone()
                labels[:prompt_len] = IGNORE_INDEX

                if labels.eq(IGNORE_INDEX).all():
                    print(prompt)
                    print("*"*20)
                    not_count+=1
                    continue

                label_tuples[label_str].append({
                    "input_ids": input_ids,
                    "attention_mask": attn_mask,
                    "labels": labels,
                    "label_str": label_str,
                    "prompt_len": prompt_len,
                    "label_id":label_id
                })

                # dataset.append({
                #     "input_ids": input_ids,
                #     "attention_mask": attn_mask,
                #     "labels": labels,
                #     "label_str": label_str,
                #     "prompt_len": prompt_len,
                #     "label_id":label_id
                # })

    if val:
        max_limit = float("inf")
        min_limit = 0
    else:
        if english:
            max_limit = 4000
            min_limit = 200
        else:
            max_limit = 4000
            min_limit = 50

    dataset = []
    
    for label, data in label_tuples.items():
        current_count = len(data)
        
        if current_count > max_limit:
            random.shuffle(data)
            dataset.extend(data[:max_limit])
            label_count[label] = min(current_count, max_limit)
            
        elif current_count < min_limit:
            rep = math.ceil(min_limit / current_count)
            rep_data = (data * rep)[:min_limit]
            dataset.extend(rep_data)
            label_count[label] = len(rep_data)
            
        else:
            dataset.extend(data)
    print(label_count)
    random.shuffle(dataset)

    print(f"unusable count = {not_count}")
    return CustomDataset(dataset), label_count

def pretrain_lora(model, train_dataloader,val_dataloaders, epochs, lr,output_path):
    global DEVICE
    device = DEVICE
    model.to(device)

    optim  = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(device)

    max_steps = (len(train_dataloader) * epochs) // ACCUMALATE_BATCHES
    warmup_steps = int(max_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, max_steps*2)

    best_val_loss = float("inf")
    global LANG_LIST
    for epoch in range(epochs):
        print(f"\nPretrain Epoch {epoch+1}/{epochs}")
        model.train()
        train_loss_total = 0

        for i, batch in enumerate(tqdm(train_dataloader), 1):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device, dtype=torch.float16):
                out  = model(input_ids, attn_mask, labels)
                loss = out.loss / ACCUMALATE_BATCHES

            scaler.scale(loss).backward()

            if i % ACCUMALATE_BATCHES == 0 or i == len(train_dataloader):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

            train_loss_total += loss.item()

            if DEBUG or DEBUG_FINE_TUNE:
                break
        avg_train_loss =  train_loss_total/len(train_dataloader)

        model.eval()

        val_loss_total_all_lang = 0

        with torch.no_grad():
            for ii, val_dataloader in enumerate(val_dataloaders):
                val_loss_total = 0
                for j, batch in enumerate(tqdm(val_dataloader), 1):
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.amp.autocast(device, dtype=torch.float16):
                        out = model.forward(input_ids, attn_mask, labels)
                        val_loss_total += out.loss.item()

                    if DEBUG or DEBUG_FINE_TUNE:
                        break

                avg_val_loss = val_loss_total / len(val_dataloader)
                perplexity = torch.exp(torch.tensor(avg_val_loss))
                val_loss_total_all_lang += avg_val_loss
                print(f"{LANG_LIST[ii+1]} => | train loss = {avg_train_loss} | val loss = {avg_val_loss} | perplexity = {perplexity}")

        if val_loss_total_all_lang < best_val_loss:
            best_val_loss = val_loss_total_all_lang
            model.lora_model.save_pretrained(os.path.join(output_path,"lora_adapter"))

        if DEBUG or DEBUG_FINE_TUNE:
            break

    # model.lora_model = model.lora_model.from_pretrained(model.lora_model.get_base_model(), "output/lora_adapter")
    return model


def train(model, train_dataloader, val_dataloaders, label2id, id2label, epochs, lr, tokenizer, loss_weights,output_path):
    global DEVICE
    device = DEVICE
    model.to(device)

    # for name, param in model.named_parameters():
    #     if "lora_" in name:
    #         param.requires_grad = True
    #         if param.dtype != torch.float32:
    #             param.data = param.data.to(torch.float32)

    loss_weights = loss_weights.to(device)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(device)

    max_steps = (len(train_dataloader) * epochs) // ACCUMALATE_BATCHES
    warmup_steps = int(max_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, max_steps)

    best_score = float("-inf")
    valid_labels = list(label2id.keys())
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
    global LANG_LIST
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}==>")
        model.train()
        total_train_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader), 1):
            with torch.amp.autocast(device, dtype=torch.float16):
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                label_id = batch["label_id"].to(device)

                out = model.forward(input_ids, attn_mask)

                pred_logits = out.logits[..., :-1, :]
                true_labels = labels[..., 1:]

                token_loss = loss_fn(pred_logits.reshape(-1, pred_logits.size(-1)), true_labels.reshape(-1))
                del out, pred_logits

                token_loss = token_loss.view(true_labels.size(0), true_labels.size(1))
                mask = (true_labels != IGNORE_INDEX).float()
                seq_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                batch_lang_weights = loss_weights[label_id]
                weighted_loss = (seq_loss * batch_lang_weights).mean()

                loss = weighted_loss / ACCUMALATE_BATCHES

            scaler.scale(loss).backward()
            del token_loss, seq_loss, loss

            if i % ACCUMALATE_BATCHES == 0 or i == len(train_dataloader):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                scheduler.step()

            total_train_loss += weighted_loss.item()

            if DEBUG: break

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()

        with torch.no_grad():
            total_val_loss_all_lang = 0
            score_all_lang=0
            for ii, val_dataloader in enumerate(val_dataloaders):

                val_preds, val_labels_list = [], []
                total_val_loss = 0
                for batch in tqdm(val_dataloader):
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    label_id = batch["label_id"].to(device)
                    plens = batch["input_lens"]

                    out = model.forward(input_ids, attn_mask)
                    pred_logits = out.logits[..., :-1, :]
                    true_labels = labels[..., 1:]

                    token_loss = loss_fn(pred_logits.reshape(-1, pred_logits.size(-1)), true_labels.reshape(-1))
                    token_loss = token_loss.view(true_labels.size(0), true_labels.size(1))

                    mask = (true_labels != IGNORE_INDEX).float()
                    seq_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                    weighted_val_loss = (seq_loss * loss_weights[label_id]).mean()
                    total_val_loss += weighted_val_loss.item()
                    max_plen = max(plens)
                    gen_input_ids = torch.stack([torch.cat([torch.full((max_plen - plens[i],), PAD_TOKEN_ID, dtype=torch.long, device=device),input_ids[i, :plens[i]]])
                        for i in range(input_ids.size(0))
                    ])

                    gen_attn_mask = torch.stack([torch.cat([torch.zeros(max_plen - plens[i], dtype=torch.long, device=device),attn_mask[i, :plens[i]]])
                        for i in range(input_ids.size(0))
                    ])

                    out_tok = model.generate(gen_input_ids, gen_attn_mask)
                    new_toks = out_tok[:, max_plen:]
                    pred_texts = tokenizer.batch_decode(new_toks, skip_special_tokens=True)

                    for ind, pred_text in enumerate(pred_texts):
                        pred_text = pred_text.strip()
                        if pred_text in label2id:
                            val_preds.append(label2id[pred_text])
                        else:
                            matches = difflib.get_close_matches(pred_text, valid_labels, n=1, cutoff=0.6)
                            val_preds.append(label2id[matches[0]] if matches else -1)

                        val_labels_list.append(label2id.get(batch["label_text"][ind], -1))


                    if DEBUG:
                        break

                avg_val_loss = total_val_loss / len(val_dataloader)
                total_val_loss_all_lang += avg_val_loss
                rel_labels = list(label2id.values())
                micro = f1_score(val_labels_list, val_preds, average="micro", labels=rel_labels, zero_division=0)
                macro = f1_score(val_labels_list, val_preds, average="macro", labels=rel_labels, zero_division=0)
                cur_score = (macro * micro) / (macro + micro + 1e-9)
                score_all_lang += cur_score

                print(f"{LANG_LIST[ii]}=> | train loss={avg_train_loss} | val loss={avg_val_loss}")
                #print(val_preds)
                print(f"micro-F1={micro} | macro-F1={macro}")
                present = sorted(set(val_labels_list + val_preds) - {-1})
                print(classification_report(
                    val_labels_list, val_preds,
                    labels=present,
                    target_names=[id2label.get(i, "unknown") for i in present],
                    zero_division=0,
                ))

        if score_all_lang > best_score:
            model.lora_model.save_pretrained(os.path.join(output_path,"lora_adapter"))
            best_score = score_all_lang


        if DEBUG:
            break

    return model


def get_dataset(tokenizer):
    if DEBUG:
        dataset_names = ["20231101.or", "20231101.tcy"]
    else:
        dataset_names = ["20231101.hi","20231101.kn","20231101.or","20231101.tcy"]
    train = []
    val = []
    num_tuples = 6144

    for dn in dataset_names:
        data = load_dataset("wikimedia/wikipedia", dn)
        dataset = create_pretrain_dataset(data,tokenizer,num_tuples)

        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        if train_len < int(num_tuples * 0.8):
            mul = 2
            print(dn,f" mul factor=> {mul}")
            train_dataset = ConcatDataset([train_dataset]*mul)
        train.append(train_dataset)
        val.append(val_dataset)

    return ConcatDataset(train), val


def get_all_label_ids():
    label2id = {}
    id2label = {}
    en = False
    global LANG_LIST
    lang_list = LANG_LIST
    lab_id = 0
    for i,lang in enumerate(lang_list):
        if i==0:
            continue
        with open(f"../sft_dataset/{lang}_map.json") as f:
            map_label = json.load(f)
            if en is False:
                for en_lab in map_label.keys():
                    label2id[en_lab]=lab_id
                    id2label[lab_id]=en_lab
                    lab_id += 1
                en = True

            # for en_lab,lang_lab in map_label.items():
            #     label2id[lang_lab] = lab_id
            #     id2label[lab_id] = lang_lab
            #     lab_id += 1

    return label2id,id2label

def custom_collate_infer(batch):
    input_ids_list = [item['input_ids'] for item in batch]
    attention_masks_list = [item['attention_mask'] for item in batch]

    articleId = [item['articleId'] for item in batch]
    sentId = [item['sentId'] for item in batch]
    sent = [item['sentence'] for item in batch]
    e1 = [item['e1'] for item in batch]
    e2 = [item['e2'] for item in batch]

    max_len = max(x.size(0) for x in input_ids_list)
    input_ids_pad = torch.stack([
        torch.cat([torch.full((max_len - x.size(0),), PAD_TOKEN_ID, dtype=torch.long), x])
        for x in input_ids_list
    ])
    attn_mask_pad = torch.stack([
        torch.cat([torch.zeros(max_len - x.size(0), dtype=torch.long), x])
        for x in attention_masks_list
    ])

    return {
        'input_ids': input_ids_pad,
        'attention_mask': attn_mask_pad,
        'articleId': articleId,
        'sentId': sentId,
        'sentence': sent,
        'e1': e1,
        'e2': e2,
        'input_len': max_len,
    }

def create_dataset_infer(data_path, tokenizer):
    dataset = []

    not_found = []

    with open(data_path, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                item = json.loads(line)
                sentence: str = normalise(item["sentText"].strip())
                article_id = item["articleId"]
                sent_id = item["sentId"]

                for rm in item["relationMentions"]:
                    e1 = normalise(rm["em1Text"])
                    e2 = normalise(rm["em2Text"])

                    if len(e2) > len(e1):
                        sentence_cur = get_marked_sent(sentence, e2, "E2")
                        if sentence_cur:
                            sentence_cur = get_marked_sent(sentence_cur, e1, "E1")
                    else:
                        sentence_cur = get_marked_sent(sentence, e1, "E1")
                        if sentence_cur:
                            sentence_cur = get_marked_sent(sentence_cur, e2, "E2")

                    if sentence_cur is None:
                        not_found.append([article_id, sent_id])
                        continue

                    prompt = [
                        {
                            "role": "system",
                            "content": "Output the relationship between the entities > format: Entity_Type/Entity_Type/Relationship_Label",
                        },
                        {
                            "role": "user",
                            "content": f"sentence: {sentence_cur}\nrelation:",
                        },
                    ]

                    prompt_str = tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    tokenized = tokenizer(
                        prompt_str,
                        truncation=True,
                        max_length=MAX_LEN,
                        return_tensors="pt",
                        padding=False,
                    )

                    input_ids = tokenized["input_ids"].squeeze(0)
                    attn_mask = tokenized["attention_mask"].squeeze(0)

                    dataset.append({
                        "input_ids": input_ids,
                        "attention_mask": attn_mask,
                        "articleId": article_id,
                        "sentId": sent_id,
                        "sentence": item["sentText"],
                        "e1": rm["em1Text"],
                        "e2": rm["em2Text"],
                    })


        print(f"Not found = {len(not_found)}")

        return CustomDataset(dataset), not_found



if __name__ == "__main__":

    mode = sys.argv[1]

    if mode == "train":
        output_path = sys.argv[2]
        os.makedirs(output_path, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        new_tokens = {"additional_special_tokens": ["<E1>", "</E1>", "<E2>", "</E2>"]}
        tokenizer.add_special_tokens(new_tokens)

        PAD_TOKEN_ID = tokenizer.eos_token_id
        label2id_en, id2label_en = get_all_label_ids()
        # print(label2id)

        max_label_tokens = 100
        model = LoraModel(len(tokenizer), max_label_tokens)

        #model = torch.compile(model)
        train_dataset_pr, val_datasets_pr = get_dataset(tokenizer)
        train_dataloader_pr = DataLoader(train_dataset_pr, PRETRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate2,
                                      num_workers=4, pin_memory=True)
        val_dataloaders_pr = []
        for data in val_datasets_pr:
            val_dataloaders_pr.append(DataLoader(data, PRETRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate2,
                                    num_workers=4, pin_memory=True))

        pretrain_lora(model,train_dataloader_pr,val_dataloaders_pr, 1,1e-4,output_path)

        del train_dataset_pr, val_datasets_pr, train_dataloader_pr, val_dataloaders_pr

        train_dataset_en, label_count_en = create_dataset(f"../en_sft_dataset/train.jsonl", tokenizer, label2id_en, True)
        val_dataset_en, _ = create_dataset("../en_sft_dataset/valid.jsonl", tokenizer, label2id_en, True, val=True)

        total_count = np.sum([c for k, c in label_count_en.items()])
        label2loss_weight = torch.sqrt(
            torch.tensor([total_count / (label_count_en[id2label_en[i]] + 1e-6) for i in range(len(label2id_en))],
                         dtype=torch.float32))
        label2loss_weight /= label2loss_weight.mean()


        # print(label2id)

        # train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=custom_collate1,
        #                               num_workers=4)
        # val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True, collate_fn=custom_collate1,
        #                             num_workers=4)

        # model = train(model, train_dataloader, val_dataloader, label2id, id2label, 5, 1e-5, tokenizer)

        # del train_dataset, train_dataloader, val_dataset, val_dataloader
        # with open("output/label_map_en.json", "w") as f:
        #     json.dump({"id2label": id2label, "label2id": label2id}, f)


        train_data = [train_dataset_en]
        val_data = [val_dataset_en]

        # all_labels = list(label2id.keys())
        label_count = label_count_en
        # label2id_label_id = {0:label2id}
        # id2label_label_id = {0:id2label}

        for i,lang  in enumerate(LANG_LIST):
            if i==0:
                continue

            with open(f"../sft_dataset/{lang}_map.json") as f:
                map_label = json.load(f)

            cur2eng = {}
            for en, cur in map_label.items():
                cur2eng[cur] = en


            # labels_cur = []
            # for key, id in label2id.items():
            #     labels_cur.append(map_label[key])
            if lang in ["tcy","or"]:
                full_dataset, label_count_cur = create_dataset(os.path.join(output_path,f"{lang}_train.jsonl"), tokenizer,
                                                               label2id_en, val=True, label_map=cur2eng)
            else:
                full_dataset, label_count_cur = create_dataset(f"../sft_dataset/{lang}_train.jsonl", tokenizer, label2id_en,val=True,label_map=cur2eng)
            # train_len = int(0.8 * len(train_dataset))
            # val_len = len(train_dataset) - train_len
            # train_dataset, val_dataset = random_split(train_dataset,[train_len,val_len])
            labels_cur = [item["label_str"] for item in full_dataset.data]
            unique_ids, counts = np.unique(labels_cur, return_counts=True)
            label_counts_full = Counter(labels_cur)

            only_one_ids = set(id for id, c in label_counts_full.items() if c == 1)

            single_ins = [i for i, l in enumerate(labels_cur) if l in only_one_ids]
            multi_inst = [i for i, l in enumerate(labels_cur) if l not in only_one_ids]
            remaining_labels = [labels_cur[i] for i in multi_inst]
            print(f"{lang} => ", single_ins)
            train_ind, val_ind = train_test_split(
                multi_inst, test_size=0.2,
                stratify=remaining_labels, random_state=1
            )
            train_ind = list(train_ind) + single_ins
            val_ind = list(val_ind) + single_ins

            final_train_cur = []
            train_dataset = CustomDataset([full_dataset.data[i] for i in train_ind])
            val_dataset = CustomDataset([full_dataset.data[i] for i in val_ind])

            label_data = defaultdict(list)
            label_count_temp = defaultdict(int)
            for data in train_dataset:
                # print(data)
                label_data[data["label_str"]].append(data)
                label_count_temp[data["label_str"]] += 1

            max_limit = 4000
            min_limit = 10
            for label, data in label_data.items():
                current_count = len(data)

                if current_count > max_limit:
                    random.shuffle(data)
                    final_train_cur.extend(data[:max_limit])
                    label_count_temp[label] = min(current_count, max_limit)

                elif current_count < min_limit:
                    rep = math.ceil(min_limit / current_count)
                    rep_data = (data * rep)[:min_limit]
                    final_train_cur.extend(rep_data)
                    label_count_temp[label] = len(rep_data)

                else:
                    final_train_cur.extend(data)

            # print("label count ",label_count_temp)
            random.shuffle(final_train_cur)

            train_dataset = CustomDataset(final_train_cur)

            target_size = 1000
            cur_size = len(train_dataset)
            rep_ratio = max(1, target_size // cur_size)
            train_dataset_rep = ConcatDataset([train_dataset]*rep_ratio)

            train_data.append(train_dataset_rep)
            val_data.append(val_dataset)

            # for k,c in label_count_cur.items():
            #     label_count[k] = c*0.8*rep_ratio

            # label2id_label_id[i+1]= label2id
            # id2label_label_id[i+1] = id2label


        # label_id = 0
        # label2id = {}
        # id2label = {}

        # for label in labels2id:
        #     label2id[label] = label_id
        #     id2label[label_id]=label
        #     label_id += 1
        #     label2loss_weight.append(label_count[label])


        train_dataset = ConcatDataset(train_data)

        train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=custom_collate1,
                                      num_workers=4, pin_memory=True)
        val_dataloaders = []
        for data in val_data:
            val_dataloaders.append(DataLoader(data, BATCH_SIZE, shuffle=True, collate_fn=custom_collate1,
                                    num_workers=4, pin_memory=True))

        model = train(model, train_dataloader, val_dataloaders, label2id_en, id2label_en, 1, 5e-5, tokenizer, label2loss_weight,output_path)

        with open(os.path.join(output_path,"label_map_en.json"), "w") as f:
            json.dump({"id2label": id2label_en, "label2id": label2id_en}, f)

        del train_dataset, train_dataloader, val_dataset, val_dataloaders
        tokenizer.save_pretrained(os.path.join(output_path,"tokenizer"))
        model.lora_model.save_pretrained(os.path.join(output_path,"lora_adapter"))

    else:
        lang = sys.argv[2]
        test_path = sys.argv[3]
        output_path = sys.argv[4]

        device = DEVICE
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_path,"tokenizer"))
        tokenizer.padding_side = "left"
        PAD_TOKEN_ID = tokenizer.eos_token_id
        label2id_all_lang, id2label_all_lang = get_all_label_ids()
        model = LoraModel(len(tokenizer), 100)

        base_model = model.lora_model.get_base_model()
        model.lora_model = PeftModel.from_pretrained(base_model, os.path.join(output_path,"lora_adapter"))
        model.lora_model = model.lora_model.half()
        model.to(device)
        model.lora_model.get_base_model().gradient_checkpointing_disable()

        test_dataset, not_found = create_dataset_infer(test_path, tokenizer)

        test_dataloader = DataLoader(test_dataset, INFER_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_infer,
                                     num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        prediction_jsonl = {}
        model.eval()

        en_labels = []
        if lang == "en":
            with open(f"../sft_dataset/hi_map.json") as f:
                label_map = json.load(f)

            label_map_temp = {}
            for en_label in label_map.keys():
                en_labels.append(en_label)
                label_map_temp[en_label] = en_label
            label_map = label_map_temp

        else:
            with open(f"../sft_dataset/{lang}_map.json") as f:
                label_map = json.load(f)

            for en_label in label_map.keys():
                en_labels.append(en_label)


        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                articleId = batch['articleId']
                sentId = batch["sentId"]
                e1_ = batch['e1']
                e2_ = batch['e2']
                sent = batch["sentence"]
                input_len = batch['input_len']

                out_tok = model.generate(input_ids, attn_mask)
                new_toks = out_tok[:, input_len:]

                pred_texts = tokenizer.batch_decode(new_toks, skip_special_tokens=True)

                for ind, pred_text in enumerate(pred_texts):
                    correct_label = ""
                    if pred_text in en_labels:
                        correct_label = label_map[pred_text]
                    else:
                        matches = difflib.get_close_matches(pred_text, en_labels, n=1, cutoff=0.6)
                        if matches:
                            correct_label = label_map[matches[0]]

                    aid = articleId[ind]
                    sid = sentId[ind]
                    exist = prediction_jsonl.get((aid, sid), None)
                    if exist:
                        exist["relationMentions"].append(
                            {"em1Text": e1_[ind], "em2Text": e2_[ind], "label": correct_label})
                    else:
                        prediction_jsonl[(aid, sid)] = {
                            "articleId": aid, "sentId": sid, "sentText": sent[ind],
                            "relationMentions": [{"em1Text": e1_[ind], "em2Text": e2_[ind], "label": correct_label}]
                        }
        os.makedirs(output_path, exist_ok=True)
        prediction_order = []

        with open(test_path, "r") as f:
            for line in tqdm(f):
                line = line.strip()
                if line:
                    item = json.loads(line)
                    aid = item["articleId"]
                    sid = item["sentId"]
                    try:
                        prediction_order.append(prediction_jsonl[(aid, sid)])
                    except:
                        for rm in item["relationMentions"]:
                            rm["label"] = ""
                        prediction_order.append(item)

        with open(os.path.join(output_path, f"Q2_{lang}.jsonl"), "w", encoding="utf-8") as f:
            for data in prediction_order:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")



