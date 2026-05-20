import sys
import os
import torch
import unicodedata
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import json
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import warnings
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import random, math
warnings.filterwarnings("ignore")
from peft import PeftModel

DEBUG = False
DEBUG_FINE_TUNE = False
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 2
PRETRAIN_BATCH_SIZE = 2
PAD_TOKEN_ID = None
ACCUMALATE_BATCHES = 16
WARMUP_RATIO = 0.05
LANG_LIST = ["en","hi","kn"]
MAX_LEN = 1024
IGNORE_INDEX = -100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
            "lang_id": lang_id
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

def pretrain_lora(model, train_dataloader,val_dataloaders, epochs, lr, output_path):
    global DEVICE
    device:str = DEVICE
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
                out  = model.lora_model(input_ids, attention_mask=attn_mask, labels=labels)
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
                        out = model.lora_model(input_ids, attention_mask=attn_mask, labels=labels)
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

def get_dataset(tokenizer):
    if DEBUG:
        dataset_names = ["20231101.or", "20231101.tcy"]
    else:
        dataset_names = ["20231101.hi","20231101.kn"]
    train = []
    val = []
    num_tuples = 8192

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
    lab_id = 0
    id2label_all_lang = []
    label2id_all_lang = []

    for i,lang in enumerate(LANG_LIST):
        if i==0:
            continue

        id2label_lang = {}
        label2id_lang = {}

        with open(f"../sft_dataset/{lang}_map.json") as f:
            map_label = json.load(f)
            if en is False:
                for key in map_label.keys():
                    label2id[key]=lab_id
                    id2label[lab_id]=key
                    lab_id += 1
                en = True
                id2label_all_lang.append(id2label)
                label2id_all_lang.append(label2id)

            for en_lab,lang_label in map_label.items():
                lab_id_cur = label2id[en_lab]
                id2label_lang[lab_id_cur] = lang_label
                label2id_lang[lang_label] = lab_id_cur

            id2label_all_lang.append(id2label_lang)
            label2id_all_lang.append(label2id_lang)

            with open(f"output/label_map_{lang}.json", "w") as f:
                json.dump({"id2label": id2label_lang, "label2id": label2id_lang}, f)

    return label2id_all_lang, id2label_all_lang

class ClassificationModel(torch.nn.Module):
    def __init__(self, tok_len, output_dim):
        super().__init__()

        qwen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.float16,device_map="auto")
        qwen_model.resize_token_embeddings(tok_len)

        qwen_model.gradient_checkpointing_enable()
        qwen_model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )

        self.lora_model= get_peft_model(qwen_model, lora_config)
        self.hidden_size = qwen_model.config.hidden_size

        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size, output_dim)
        )

    def forward(self, input_ids, attn_mask, e1_inds,e2_inds,eos_inds):
        outputs = self.lora_model.base_model.model.model(input_ids, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=input_ids.device)

        e1 = hidden[batch_indices, e1_inds, :]
        e2 = hidden[batch_indices, e2_inds, :]
        eos = hidden[batch_indices, eos_inds, :]

        inp = torch.concat([e1, e2, eos], dim=1).to(torch.float32)
        return self.head(inp)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id]


def custom_collate_function(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    e1_starts = [torch.tensor(item['entity_ind'][0]) for item in batch]
    e1_ends = [torch.tensor(item['entity_ind'][1]) for item in batch]
    e2_starts = [torch.tensor(item['entity_ind'][2]) for item in batch]
    e2_ends = [torch.tensor(item['entity_ind'][3]) for item in batch]
    eos_indices = [torch.tensor(item['entity_ind'][4]) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]


    input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_masks_pad = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_pad,
        'attention_mask': attention_masks_pad,
        'e1_sind': torch.stack(e1_starts),
        'e1_eind': torch.stack(e1_ends),
        'e2_sind': torch.stack(e2_starts),
        'e2_eind': torch.stack(e2_ends),
        'eos_ind': torch.stack(eos_indices),
        'labels': torch.stack(labels)
    }


def custom_collate_infer(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    e1_starts = [torch.tensor(item['entity_ind'][0]) for item in batch]
    e1_ends = [torch.tensor(item['entity_ind'][1]) for item in batch]
    e2_starts = [torch.tensor(item['entity_ind'][2]) for item in batch]
    e2_ends = [torch.tensor(item['entity_ind'][3]) for item in batch]
    eos_indices = [torch.tensor(item['entity_ind'][4]) for item in batch]
    articleId = [item['articleId'] for item in batch]
    sentId = [item['sentId'] for item in batch]
    sent = [item['sentence'] for item in batch]
    e1 = [item['e1'] for item in batch]
    e2 = [item['e2'] for item in batch]

    input_ids_pad = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_masks_pad = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_pad,
        'attention_mask': attention_masks_pad,
        'e1_sind': torch.stack(e1_starts),
        'e1_eind': torch.stack(e1_ends),
        'e2_sind': torch.stack(e2_starts),
        'e2_eind': torch.stack(e2_ends),
        'eos_ind': torch.stack(eos_indices),
        'articleId':articleId,
        'sentId':sentId,
        "e1":e1,
        "e2":e2,
        "sentence":sent
    }

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



def create_dataset(data_path, tokenizer, label2id, lang_id = 0, val=False):

    e1s = tokenizer.convert_tokens_to_ids("<E1>")
    e1e = tokenizer.convert_tokens_to_ids("</E1>")
    e2s = tokenizer.convert_tokens_to_ids("<E2>")
    e2e = tokenizer.convert_tokens_to_ids("</E2>")
    eos = tokenizer.eos_token_id

    label_count = defaultdict(int)
    label_data = defaultdict(list)
    not_count = 0

    with open(data_path, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                item = json.loads(line)
                sentence: str = normalise(item["sentText"].strip())
                for rm in item["relationMentions"]:
                    e1 = normalise(rm["em1Text"])
                    e2 = normalise(rm["em2Text"])
                    try:
                        label_id_cur = label2id[rm["label"]]
                        label_count[rm["label"]] += 1
                    except:
                        not_count+=1
                        continue

                    # print("*"*30)
                    # print(e1,"--",e2,"--",sentence)

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

                    tokenized = tokenizer(
                        sentence_cur,
                        truncation=True,
                        return_tensors="pt",
                        max_length=MAX_LEN,
                        padding = False,
                    )
                    sentence_ids = tokenized["input_ids"].squeeze(0)
                    attention_mask = tokenized["attention_mask"].squeeze(0)
                    try:
                        e1_ind = [(sentence_ids == e1s).nonzero(as_tuple=True)[0][0].item(), (sentence_ids == e1e).nonzero(as_tuple=True)[0][0].item()]
                        e2_ind = [(sentence_ids == e2s).nonzero(as_tuple=True)[0][0].item(), (sentence_ids == e2e).nonzero(as_tuple=True)[0][0].item()]
                        eos_ind = [len(sentence_ids) - 1]
                        label_data[label_id_cur].append({"input_ids": sentence_ids,"attention_mask":attention_mask, "labels":label_id_cur,
                                     "entity_ind": e1_ind + e2_ind + eos_ind,"lang_id":lang_id })
                        # dataset.append({"input_ids": sentence_ids,"attention_mask":attention_mask, "labels":label_id_cur,
                        #             "entity_ind": e1_ind + e2_ind + eos_ind,"lang_id":lang_id })
                    except:
                        not_count+=1
        print(f"unusable count = {not_count}")

        if val:
            max_limit = float("inf")
            min_limit = 0
        else:
            if lang_id ==0:
                max_limit = 3000
                min_limit = 200
            else:
                max_limit = 3000
                min_limit = 50

        dataset = []

        for label, data in label_data.items():
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
        # print("create dataset =>",label_count)
        random.shuffle(dataset)

        return CustomDataset(dataset), label_count


def create_dataset_infer(data_path, tokenizer):
    dataset = []
    e1s = tokenizer.convert_tokens_to_ids("<E1>")
    e1e = tokenizer.convert_tokens_to_ids("</E1>")
    e2s = tokenizer.convert_tokens_to_ids("<E2>")
    e2e = tokenizer.convert_tokens_to_ids("</E2>")
    eos = tokenizer.eos_token_id

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
                    # print("*"*30)
                    # print(e1,"--",e2,"--",sentence)
                    # print(sentence,e1,e2)
                    if len(e2) > len(e1):
                        sentence_cur = get_marked_sent(sentence, e2, "E2")
                        if sentence_cur:
                            sentence_cur = get_marked_sent(sentence_cur, e1, "E1")
                    else:
                        sentence_cur = get_marked_sent(sentence, e1, "E1")
                        if sentence_cur:
                            sentence_cur = get_marked_sent(sentence_cur, e2, "E2")

                    if sentence_cur is None:

                        not_found.append([article_id,sent_id])
                        continue

                    tokenized = tokenizer(
                        sentence_cur,
                        truncation=True,
                        return_tensors="pt",
                        max_length=MAX_LEN,
                        padding = False,
                    )
                    sentence_ids = tokenized["input_ids"].squeeze(0)
                    attention_mask = tokenized["attention_mask"].squeeze(0)
                    try:
                        e1_ind = [(sentence_ids == e1s).nonzero(as_tuple=True)[0][0].item(), (sentence_ids == e1e).nonzero(as_tuple=True)[0][0].item()]
                        e2_ind = [(sentence_ids == e2s).nonzero(as_tuple=True)[0][0].item(), (sentence_ids == e2e).nonzero(as_tuple=True)[0][0].item()]
                        eos_ind = [len(sentence_ids) - 1]
                        dataset.append({"input_ids": sentence_ids,"attention_mask":attention_mask, "sentence":item["sentText"],"e1":rm["em1Text"], "e2":rm["em2Text"],
                                    "entity_ind": e1_ind + e2_ind + eos_ind, "articleId":article_id, "sentId":sent_id})
                    except:
                        not_found.append([article_id, sent_id])
                        continue

        print(f"Not foun = {len(not_found)}")

        return CustomDataset(dataset), not_found


def train(model, train_dataloader, val_dataloaders, loss_weight, label2idlist, id2labellist, epochs, lr, output_path):
    global DEVICE
    device = DEVICE
    model.to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight,dtype=torch.float32).to(device))

    scaler = torch.amp.GradScaler(device)

    max_steps = (len(train_dataloader) * epochs)//ACCUMALATE_BATCHES
    warmup_steps = int(max_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps*2
    )
    best_score = float("-inf")
    global LANG_LIST
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}==>")
        model.train()
        total_train_loss = 0
        i=0
        for batch in tqdm(train_dataloader):
            i+=1
            with torch.amp.autocast(device, dtype=torch.float16):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                e1 = batch['e1_eind'].to(device)
                e2 = batch['e2_eind'].to(device)
                eos = batch['eos_ind'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, mask, e1, e2, eos)
                loss = loss_fn(logits, labels)
                loss = loss / ACCUMALATE_BATCHES
                scaler.scale(loss).backward()

                if i % ACCUMALATE_BATCHES == 0 or i == len(train_dataloader):
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                    scheduler.step()

            total_train_loss += loss.item()
            if DEBUG:
                break
        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()

        total_val_loss_all_lang = 0
        total_score = 0

        with torch.no_grad():
            for ii, val_dataloader in enumerate(val_dataloaders):
                total_val_loss = 0
                val_preds = []
                val_labels = []
                label2id = label2idlist[ii]
                id2label = id2labellist[ii]

                for batch in tqdm(val_dataloader):
                    input_ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    e1 = batch['e1_eind'].to(device)
                    e2 = batch['e2_eind'].to(device)
                    eos = batch['eos_ind'].to(device)
                    labels = batch['labels'].to(device)

                    logits = model(input_ids, mask, e1, e2, eos)
                    loss = loss_fn(logits, labels)
                    total_val_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1)

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch['labels'].numpy())
                    if DEBUG:
                        break
                avg_val_loss = total_val_loss/len(val_dataloader)
                total_val_loss_all_lang += avg_val_loss

                rel_labels = [v for k, v in label2id.items()]
                micro = f1_score(val_labels,val_preds, average='micro', labels=rel_labels)
                macro = f1_score(val_labels,val_preds, average='macro', labels=rel_labels)

                cur_score = (macro * micro) / (macro + micro + 1e-9)
                total_score += cur_score

                print(f"{LANG_LIST[ii]}=> | train loss= {avg_train_loss} | val loss= {avg_val_loss}")
                print(f"micro-F1: {micro} | macro-F1: {macro}")
                print(classification_report(val_labels, val_preds, labels=list(sorted(id2label.keys())),
                                            target_names=list(id2label.values())))
        if total_score >  best_score:
            model.lora_model.save_pretrained(os.path.join(output_path,"lora_adapters"))
            torch.save(model.head.state_dict(), os.path.join(output_path,"classifier_head.pth"))
            best_score = total_score

        if DEBUG:
            break

    # model.head.load_state_dict(torch.load("output/classifier_head.pth", map_location=device))
    # base_model = model.lora_model.get_base_model()
    # model.lora_model = PeftModel.from_pretrained(base_model, "output/lora_adapters")
    # model.lora_model = model.lora_model.half()
    return model


if __name__ == "__main__":

    mode = sys.argv[1]

    if mode == "train":
        output_path = sys.argv[2]
        os.makedirs(output_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        new_tokens = {'additional_special_tokens': ['<E1>', '</E1>', '<E2>', '</E2>']}
        tokenizer.add_special_tokens(new_tokens)

        PAD_TOKEN_ID = tokenizer.eos_token_id
        label2id_all_lang, id2label_all_lang = get_all_label_ids()
        model = ClassificationModel(len(tokenizer), len(label2id_all_lang[0]))
        train_dataset_pr, val_datasets_pr = get_dataset(tokenizer)
        train_dataloader_pr = DataLoader(train_dataset_pr, PRETRAIN_BATCH_SIZE, shuffle=True,
                                         collate_fn=custom_collate2,
                                         num_workers=4, pin_memory=True)
        val_dataloaders_pr = []
        for data in val_datasets_pr:
            val_dataloaders_pr.append(DataLoader(data, PRETRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate2,
                                                 num_workers=4, pin_memory=True))

        pretrain_lora(model, train_dataloader_pr, val_dataloaders_pr, 1, 1e-4, output_path)

        del train_dataset_pr, val_datasets_pr, train_dataloader_pr, val_dataloaders_pr

        train_dataset_en, label_count = create_dataset("../en_sft_dataset/train.jsonl", tokenizer, label2id_all_lang[0],
                                                       lang_id=0)
        val_dataset_en, _ = create_dataset("../en_sft_dataset/valid.jsonl", tokenizer, label2id_all_lang[0], lang_id=0,
                                           val=True)

        num_classes = len(label2id_all_lang[0])

        total_count = np.sum([c for k, c in label_count.items()])
        label2loss_weight = torch.sqrt(torch.tensor(
            [total_count / (label_count[id2label_all_lang[0][i]] + 1e-6) for i in range(len(id2label_all_lang[0]))],
            dtype=torch.float32))
        label2loss_weight /= label2loss_weight.mean()

        # print(label2id)

        # train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=custom_collate_function,
        #                               num_workers=4)
        # val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True, collate_fn=custom_collate_function,
        #                             num_workers=4)



        # model = train(model, train_dataloader, val_dataloader, loss_weight, label2id, id2label,3,5e-5, tokenizer)

        # del train_dataset, train_dataloader, val_dataset, val_dataloader
        with open(os.path.join(output_path,"label_map_task1.json"), "w") as f:
            json.dump({"id2labellist": id2label_all_lang, "label2idlist": label2id_all_lang}, f)

        with open(os.path.join(output_path,"label_map_en.json"), "w") as f:
            json.dump({"id2label": id2label_all_lang[0], "label2id": label2id_all_lang[0]}, f)


        train_data = [train_dataset_en]
        val_data = [val_dataset_en]

        for i,lang  in enumerate(LANG_LIST):
            if i==0:
                continue
            full_dataset, label_count_cur = create_dataset(f"../sft_dataset/{lang}_train.jsonl", tokenizer,label2id_all_lang[i],i,True)
            # train_len = int(0.8 * len(train_dataset))
            # val_len = len(train_dataset) - train_len
            # train_dataset, val_dataset = random_split(train_dataset,[train_len,val_len])

            labels_cur = [item["labels"] for item in full_dataset.data]
            unique_ids, counts = np.unique(labels_cur, return_counts=True)

            label_counts_full = Counter(labels_cur)
            only_one_ids = set(id for id, c in label_counts_full.items() if c == 1)

            single_ins = [i for i, l in enumerate(labels_cur) if l in only_one_ids]
            multi_inst = [i for i, l in enumerate(labels_cur) if l not in only_one_ids]
            remaining_labels = [labels_cur[i] for i in multi_inst]
            print(f"{lang} => ",single_ins)
            train_ind, val_ind = train_test_split(
                multi_inst, test_size=0.2,
                stratify=remaining_labels, random_state=1
            )
            train_ind = list(train_ind) + single_ins
            val_ind = list(val_ind) + single_ins
            final_train_cur = []

            train_dataset = [full_dataset.data[i] for i in train_ind]
            val_dataset = CustomDataset([full_dataset.data[i] for i in val_ind])

            label_data = defaultdict(list)
            label_count_cur = defaultdict(int)
            for data in train_dataset:
                # print(data)
                label_data[data["labels"]].append(data)
                label_count_cur[data["labels"]]+=1

            max_limit = 3000
            min_limit = 20
            for label, data in label_data.items():
                current_count = len(data)

                if current_count > max_limit:
                    random.shuffle(data)
                    final_train_cur.extend(data[:max_limit])
                    label_count_cur[label] = min(current_count, max_limit)

                elif current_count < min_limit:
                    rep = math.ceil(min_limit / current_count)
                    rep_data = (data * rep)[:min_limit]
                    final_train_cur.extend(rep_data)
                    label_count_cur[label] = len(rep_data)

                else:
                    final_train_cur.extend(data)

            # print("label count ",label_count)
            random.shuffle(final_train_cur)

            train_dataset = CustomDataset(final_train_cur)
            # print("len train",len(train_dataset))
            # print("len val",len(val_dataset))
            target_size = 2000
            cur_size = len(train_dataset)
            rep_ratio = max(1, target_size // cur_size)
            train_dataset_rep = ConcatDataset([train_dataset] * rep_ratio)
            train_data.append(train_dataset_rep)
            val_data.append(val_dataset)
            # print("len train", len(train_dataset_rep))

            #
            # with open(f"../sft_dataset/{lang}_map.json") as f:
            #     map_label = json.load(f)
            #
            # label2id_cur = {}
            # id2label_cur = {}
            #
            # for key, id in label2id.items():
            #     label2id_cur[map_label[key]] = id
            #     id2label_cur[id] = map_label[key]
            #

        train_dataset = ConcatDataset(train_data)


        train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=custom_collate_function,
                                      num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        val_dataloaders = []
        for data in val_data:
            val_dataloaders.append(DataLoader(data, BATCH_SIZE, shuffle=True, collate_fn=custom_collate_function,
                                    num_workers=4,prefetch_factor=2, persistent_workers=True))

        model = train(model, train_dataloader, val_dataloaders, label2loss_weight, label2id_all_lang, id2label_all_lang, 1, 5e-5, output_path)

        tokenizer.save_pretrained(os.path.join(output_path,"tokenizer"))
        model.lora_model.save_pretrained(os.path.join(output_path,"lora_adapters"))

    else:
        lang = sys.argv[2]
        test_path = sys.argv[3]
        output_path = sys.argv[4]

        device = DEVICE
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_path,"tokenizer"))

        PAD_TOKEN_ID = tokenizer.eos_token_id
        label2id_all_lang, id2label_all_lang = get_all_label_ids()
        model = ClassificationModel(len(tokenizer), len(label2id_all_lang[0]))
        model.head.load_state_dict(torch.load(os.path.join(output_path,"classifier_head.pth"), map_location=device))
        base_model = model.lora_model.get_base_model()
        model.lora_model = PeftModel.from_pretrained(base_model, os.path.join(output_path,"lora_adapters"))
        model.lora_model = model.lora_model.half()
        model.to(device)
        test_dataset, not_found = create_dataset_infer(test_path,tokenizer)

        test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=custom_collate_infer,
                                      num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        prediction_jsonl = {}
        model.eval()
        with open(os.path.join(output_path,"label_map_en.json"), "r") as f:
            label_map = json.load(f)["id2label"]

        if lang == "en":
            with open(f"../sft_dataset/hi_map.json", "r") as f:
                temp = json.load(f)
                en2cur = {}
                for key in temp.keys():
                    en2cur[key] = key
        else:
            with open(f"../sft_dataset/{lang}_map.json", "r") as f:
                en2cur = json.load(f)

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                e1 = batch['e1_eind'].to(device)
                e2 = batch['e2_eind'].to(device)
                eos = batch['eos_ind'].to(device)
                articleId = batch['articleId']
                sentId = batch["sentId"]
                e1_ = batch['e1']
                e2_ = batch['e2']
                sent = batch["sentence"]

                logits = model(input_ids, mask, e1, e2, eos)
                preds = torch.argmax(logits, dim=-1)

                i=0
                for aid,sid in zip(articleId,sentId):
                    exist = prediction_jsonl.get((aid,sid),None)
                    try:
                        cur_label = en2cur[label_map[str(preds[i].item())]]
                    except Exception as e:
                        print(str(e))
                        cur_label = ""
                    if exist:
                        exist["relationMentions"].append({"em1Text": e1_[i] , "em2Text": e2_[i], "label": cur_label })
                    else:
                        prediction_jsonl[(aid,sid)] = {"articleId": aid, "sentId": sid, "sentText": sent[i],
                                                       "relationMentions": [ { "em1Text": e1_[i], "em2Text": e2_[i], "label": cur_label } ]}
                    i+=1

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
                        prediction_order.append(prediction_jsonl[(aid,sid)])
                    except:
                        for rm in item["relationMentions"]:
                            rm["label"] = ""
                        prediction_order.append(item)

        with open(os.path.join(output_path, f"Q1_{lang}.jsonl"), "w", encoding="utf-8") as f:
            for data in prediction_order:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
