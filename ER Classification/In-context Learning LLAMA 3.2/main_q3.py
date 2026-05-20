from vllm import LLM, SamplingParams
import faiss
import sys
import json
import warnings
import numpy as np
import pickle
import gc
from tqdm import tqdm
import difflib
import unicodedata
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time, random, math

LANG_LIST = ["hi","kn","or","tcy"]
warnings.filterwarnings("ignore")

DEBUG = False
MODEL_NAME = "/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FaissDatabase:
    def __init__(self, nlist= 512):
        global DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        self.model = AutoModel.from_pretrained("google/muril-base-cased").to(DEVICE)
        self.dim = 768
        self.quantizer = faiss.IndexFlatIP(self.dim)
        self.en_index = faiss.IndexIVFFlat(self.quantizer,self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.en_index.nprobe = 20
        self.lang_index = [faiss.IndexFlatIP(self.dim) for _ in range(len(LANG_LIST))]
        self.examples = []
        self.examples_lang = [[],[],[],[]]
        self.index_len_lang = [0,0,0,0]
        self.index_len = 0
        self.k_en = 5
        self.k_lang = 4

    def insert_en(self, emb, inp_str, lab_str):
        faiss.normalize_L2(emb)
        self.en_index.train(emb)
        self.en_index.add(emb)

        for i in range(len(inp_str)):
            self.examples.append([inp_str[i], lab_str[i]])
            self.index_len += 1

    def insert_other(self,emb, inp_str, lab_str, lang):
        ind  = LANG_LIST.index(lang)
        faiss.normalize_L2(emb)
        self.lang_index[ind].add(emb)

        for i in range(len(inp_str)):
            self.examples_lang[ind].append( [inp_str[i], lab_str[i]])
            self.index_len_lang[ind] += 1


    def retrieve(self, query, lang_ind):
        faiss.normalize_L2(query)
        _ , ind_en = self.en_index.search(query, self.k_en)
        _ , ind_lang = self.lang_index[lang_ind].search(query,self.k_lang)
        # ind_other = {}
        # for i in range(len(LANG_LIST)):
        #     if i!=lang_ind:
        #         _, ind_cur = self.lang_index[i].search(query,1)
        #         ind_other[i] = ind_cur

        batch_res = []
        for i in range(len(query)):
            res = []

            for ind in ind_en[i]:
                if ind != -1:
                    res.append(self.examples[ind])

            for ind in ind_lang[i]:
                if ind != -1:
                    res.append(self.examples_lang[lang_ind][ind])
            # for ii in ind_other.keys():
            #     for ind in ind_other[ii][i]:
            #         if ind != -1:
            #             res.append(self.examples_lang[ii][ind])
            batch_res.append(res)

        return batch_res

    def get_embeddings(self, texts):
        all_embeddings = []
        batch_size = BATCH_SIZE
        global DEVICE
        self.model.to(DEVICE)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128).to(
                DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                emb = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy())
        self.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

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

def create_datalist(path):
    label_count = defaultdict(int)
    label_data = defaultdict(list)

    with open(path,"r") as f:
        for line in f:
            data = json.loads(line.strip())
            sentence = normalise(data["sentText"].strip())
            for rm in data["relationMentions"]:
                e1 = normalise(rm["em1Text"])
                e2 = normalise(rm["em2Text"])
                label = normalise(rm["label"])

                if len(e2) > len(e1):
                    sentence_cur = get_marked_sent(sentence, e2, "E2")
                    if sentence_cur:
                        sentence_cur = get_marked_sent(sentence_cur, e1, "E1")
                else:
                    sentence_cur = get_marked_sent(sentence, e1, "E1")
                    if sentence_cur:
                        sentence_cur = get_marked_sent(sentence_cur, e2, "E2")

                if sentence_cur is not None:
                    label_data[label].append({"prompt": sentence_cur, "label": label})
                    label_count[label] += 1

    max_limit = 1000

    dataset = []

    for label, data in label_data.items():
        current_count = len(data)

        if current_count > max_limit:
            random.shuffle(data)
            dataset.extend(data[:max_limit])
            label_count[label] = min(current_count, max_limit)
        else:
            dataset.extend(data)
    print(label_count)
    random.shuffle(dataset)

    return dataset, label_count

def create_datalist_infer(path):
    dataset = []

    with open(path,"r") as f:
        for line in f:
            data = json.loads(line.strip())
            sentence = normalise(data["sentText"].strip())
            for rm in data["relationMentions"]:
                e1 = normalise(rm["em1Text"])
                e2 = normalise(rm["em2Text"])
                label = normalise(rm["label"])

                if len(e2) > len(e1):
                    sentence_cur = get_marked_sent(sentence, e2, "E2")
                    if sentence_cur:
                        sentence_cur = get_marked_sent(sentence_cur, e1, "E1")
                else:
                    sentence_cur = get_marked_sent(sentence, e1, "E1")
                    if sentence_cur:
                        sentence_cur = get_marked_sent(sentence_cur, e2, "E2")

                if sentence_cur is not None:
                    dataset.append({"prompt": sentence_cur, "label": label,
                                    "sentence": data["sentText"],
                                    "e1": rm["em1Text"],
                                    "e2": rm["em2Text"],
                                    "articleId":data["articleId"],
                                    "sentId":data["sentId"]})

    return dataset


def get_all_label_ids():
    label2id = {}
    id2label = {}
    global LANG_LIST

    lab_id = 0

    with open(f"../sft_dataset/hi_map.json") as f:
        map_label = json.load(f)
        for en_lab in map_label.keys():
            label2id[en_lab] = lab_id
            id2label[lab_id] = en_lab
            lab_id += 1


            # for en_lab, lang_lab in map_label.items():
            #     label2id[lang_lab] = lab_id
            #     id2label[lab_id] = lang_lab
            #     lab_id += 1

    return label2id, id2label


def run_eval_vllm(llm, data, database, training=True, all_label_map=None, en_labels=None, lang_ind=0, en2cur=None):
    test_keys = [item["prompt"] for item in data]

    if training:
        labels_true = [all_label_map[item["label"]] for item in data]

    print(f"{len(test_keys)} queries")
    t1 = time.time()
    # outputs = llm.encode(test_keys)
    hidden_embs = database.get_embeddings(test_keys)
    examples_all = database.retrieve(hidden_embs, lang_ind)

    prompts = []
    for i, examples in enumerate(examples_all):

        messages = [{"role": "system", "content": f'''You will be given a sentence and entities within it will be marked with <E1> entity_1 </E1> and <E2> entity_2 </E2>,
                                                      You have to predict entity types and relationship 
                                                      You have to choose the label from the following set of valid labels {en_labels}'''}]
        
        for (sent, label) in examples:
            messages.append({"role": "user", "content": f"{sent}\nLabel:"})
            messages.append({"role": "assistant", "content": label})

        messages.append({"role": "user", "content": f"{test_keys[i]}\nLabel:"})

        prompt = llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100, stop=["\n"])
    print("generating")
    outputs = llm.generate(prompts, sampling_params)
    print("completed")
    preds = [output.outputs[0].text.strip() for output in outputs]
    pred_cur_lang = []
    for ind, pred_text in enumerate(preds):
        correct_label = ""
        #print(pred_text)
        if pred_text in en_labels:
            correct_label = en2cur[pred_text]
        else:
            matches = difflib.get_close_matches(pred_text, en_labels, n=1, cutoff=0.6)
            if matches:
                correct_label = en2cur[matches[0]]
        pred_cur_lang.append(correct_label)

    time_taken = (time.time() - t1)/60

    if training:
        macro = f1_score(labels_true, preds, average='macro', zero_division=0)
        micro = f1_score(labels_true, preds, average='micro', zero_division=0)
        print(f"time taken => {time_taken} mins")
        print(f"{LANG_LIST[lang_ind]} =>>")
        print(f"macro F1: {macro} | micro: {micro}")
        print(classification_report(labels_true, preds, zero_division=0))
    else:
        return pred_cur_lang


if __name__ == "__main__":

    lang_mode = sys.argv[1]
    test_file = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)
    database = FaissDatabase()
    label2id, id2label = get_all_label_ids()
    en_labels = list(label2id.keys())
    print(en_labels)
    exist = False
    try:
        if os.path.exists(os.path.join(output_dir, "faiss_en_index.bin")):
            database.en_index = faiss.read_index(os.path.join(output_dir,"faiss_en_index.bin"))
            with open(os.path.join(output_dir,"faiss_config.pkl"), "rb") as f:
                config = pickle.load(f)
                database.examples = config["examples"]
                database.index_len = config["len"]

            for i,lang in enumerate(LANG_LIST):
                database.lang_index[i] = faiss.read_index(os.path.join(output_dir, f"faiss_{lang}_index.bin"))
                with open(os.path.join(output_dir, f"faiss_config_{lang}.pkl"), "rb") as f:
                    config = pickle.load(f)
                    database.examples_lang[i] = config["examples"]

            with open(os.path.join(output_dir,"all_label_map.json")) as f:
                all_label_map = json.load(f)


            exist = True
    except:
        print("couldn't load, will retrain index")

    if not exist:
        database = FaissDatabase()
        en_data, _ = create_datalist("../en_sft_dataset/train.jsonl")
        val_datasets = []

        all_label_map = {label: label for label in label2id.keys()}
        all_label_map['/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನ'] = "None"

        train_texts = [item["prompt"] for item in en_data]
        train_labels = [all_label_map[item["label"]] for item in en_data]

        hidden_embs = database.get_embeddings(train_texts)
        database.insert_en(hidden_embs, train_texts, train_labels)
        del hidden_embs

        for i, lang in enumerate(LANG_LIST):
            cur_dataset, label_count_cur = create_datalist(f"../sft_dataset/{lang}_train.jsonl")

            with open(f"../sft_dataset/{lang}_map.json") as f:
                map_label = json.load(f)
            cur2eng = {}

            for en, cur in map_label.items():
                all_label_map[cur] = en
            labels_cur = [item["label"] for item in cur_dataset]

            # unique_ids, counts = np.unique(labels_cur, return_counts=True)
            # only_one_ids = unique_ids[counts == 1]
            #
            # repeat_data = []
            # for item in cur_dataset:
            #     if item["label"] in only_one_ids:
            #         repeat_data.append(item)
            #
            # cur_dataset.extend(repeat_data)

            # labels_cur = [item["label"] for item in cur_dataset]

            # train_ind, val_ind = train_test_split(
            #     range(len(cur_dataset)),
            #     test_size=0.2,
            #     stratify=labels_cur,
            #     random_state=1
            # )
            # train_data_cur = [cur_dataset[i] for i in train_ind]
            train_data_cur = cur_dataset
            train_texts = [item["prompt"] for item in train_data_cur]
            train_labels = [all_label_map[item["label"]] for item in train_data_cur]
            ind = LANG_LIST.index(lang)
            hidden_embs = database.get_embeddings(train_texts)
            database.insert_other(hidden_embs, train_texts, train_labels,lang)
            del hidden_embs
            # val_data_cur = [cur_dataset[i] for i in val_ind]
            # val_datasets.append(val_data_cur)

        database.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        print("trained faiss index")
        llm = LLM(model=MODEL_NAME,
                  tokenizer=MODEL_NAME, gpu_memory_utilization=0.9, enforce_eager=True, dtype="half",
                  max_model_len=8192, revision="main",disable_custom_all_reduce=True, skip_tokenizer_init=False)

        # for i,val_data in enumerate(val_datasets):
        #     with open(f"../sft_dataset/{LANG_LIST[i]}_map.json") as f:
        #         en2cur_map = json.load(f)
        #     run_eval_vllm(llm,val_data,database,True, all_label_map, en_labels,i,en2cur_map)

        with open(os.path.join(output_dir,"all_label_map.json"), "w") as f:
            json.dump(all_label_map, f)

        os.makedirs("output", exist_ok=True)
        faiss.write_index(database.en_index,os.path.join(output_dir,"faiss_en_index.bin"))
        with open(os.path.join(output_dir,"faiss_config.pkl"),"wb") as f:
            pickle.dump({"examples":database.examples, "len":database.index_len},f)

        for i,lang in enumerate(LANG_LIST):
            faiss.write_index(database.lang_index[i], os.path.join(output_dir,f"faiss_{lang}_index.bin"))
            with open(os.path.join(output_dir,f"faiss_config_{lang}.pkl"), "wb") as f:
                pickle.dump({"examples": database.examples_lang[i], "len": database.index_len_lang[i]}, f)
    else:
        database.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        llm = LLM(model=MODEL_NAME,
                  tokenizer=MODEL_NAME, gpu_memory_utilization=0.9, enforce_eager=True, dtype="half",
                  max_model_len=8192, revision="main", disable_custom_all_reduce=True, skip_tokenizer_init=False)


    with open(f"../sft_dataset/{lang_mode}_map.json") as f:
        en2cur_map = json.load(f)

    with open(os.path.join(output_dir,"all_label_map.json")) as f:
        all_label_map = json.load(f)

    test_dataset = create_datalist_infer(test_file)
    prediction = run_eval_vllm(llm,test_dataset,database,False,all_label_map, en_labels,LANG_LIST.index(lang_mode),en2cur_map)
    prediction_jsonl = {}

    for ind, pred_text in enumerate(prediction):
        aid = test_dataset[ind]["articleId"]
        sid = test_dataset[ind]["sentId"]
        e1 = test_dataset[ind]["e1"]
        e2 = test_dataset[ind]["e2"]
        sent = test_dataset[ind]["sentence"]
        exist = prediction_jsonl.get((aid, sid), None)
        if exist:
            exist["relationMentions"].append(
                {"em1Text": e1, "em2Text": e2, "label": pred_text})
        else:
            prediction_jsonl[(aid, sid)] = {
                "articleId": aid, "sentId": sid, "sentText": sent,
                "relationMentions": [{"em1Text": e1, "em2Text": e2, "label": pred_text}]
            }

    os.makedirs(output_dir, exist_ok=True)
    prediction_order = []

    with open(test_file , "r") as f:
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

    with open(os.path.join(output_dir, f"Q3_{lang_mode}.jsonl"), "w", encoding="utf-8") as f:
        for data in prediction_order:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")



