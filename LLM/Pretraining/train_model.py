
import os.path
import torch
from partb.bpe_tokenizer import BPETokenizer
from parta.model import *
import wandb
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 30

class CustomDataset(Dataset):
    def __init__(self, x):
        self.data = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x_tensor = self.data[item][:-1].clone()
        y_tensor = self.data[item][1:].clone()
        return {
            "input_ids": x_tensor,
            "attention_mask": torch.ones_like(x_tensor),
            "labels": y_tensor
        }

def create_dataset(ttext):
    x = []
    for line in ttext:
        if len(line) > MAX_LEN:
            for ii in range(0, len(line), MAX_LEN):
                x.append(torch.tensor(line[ii:ii + MAX_LEN]))
        else:
            x.append(torch.tensor(line))
    return CustomDataset(x)

def main(args):
    with torch.no_grad():
        tokenizer: BPETokenizer = BPETokenizer()
        tokenizer.load(args.tokenizer_path)
        train_ttext = []

        with open(args.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                train_ttext.append(tokenizer.encode(line))

        valid_ttext = []
        total_valid_chars = 0
        with open(args.valid_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                total_valid_chars += len(line)
                valid_ttext.append(tokenizer.encode(line))

        train_dataset = create_dataset(train_ttext)
        valid_dataset = create_dataset(valid_ttext)
        config = {
            "d_model": 256,
            "n_heads": 8,
            "d_head": 32,
            "n_layers": 10,
            "vocab_size": tokenizer.get_vocab_size(),
            "mode": "tanh-clipped",
            "tau": 0.5
        }

        train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,num_workers=4, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False,num_workers=4, collate_fn=collate_fn)

    model = LanguageModel(config, device=DEVICE)
    train_loss = []
    valid_loss = []
    bpc_all = []
    loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)

    best_valid_loss = float('inf')

    warmup_step = int(len(train_dataloader)*EPOCHS*0.1)
    t_max = int(len(train_dataloader)*EPOCHS) - warmup_step

    linear_sch = LinearLR(optim, start_factor=0.1, total_iters=warmup_step)
    cosine_sch = CosineAnnealingLR(optim, T_max=t_max)
    scheduler = SequentialLR(optim, schedulers=[linear_sch, cosine_sch], milestones=[warmup_step])

    for e in range(EPOCHS):
        print(f"Epoch {e+1}/{EPOCHS}==>")
        total_train_loss = 0.0
        total_tokens_train = 0
        model.train()
        for x, attn_mask, y in train_dataloader:
            x, attn_mask, y = x.to(DEVICE), attn_mask.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            logits = model(x,attn_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            with torch.no_grad():
                total_train_loss += loss.item()
                total_tokens_train += (y != 0).sum().item()
            scheduler.step()

        with torch.no_grad():
            total_valid_loss = 0.0
            total_tokens_valid = 0
            model.eval()
            for x, attn_mask, y in valid_dataloader:
                x, attn_mask, y = x.to(DEVICE), attn_mask.to(DEVICE), y.to(DEVICE)
                logits = model(x,attn_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                total_valid_loss += loss.item()
                total_tokens_valid += (y != 0).sum().item()

            bpc = total_valid_loss / (total_valid_chars * torch.log(torch.tensor(2)).item())
            bpc_all.append(bpc)
            valid_loss.append(total_valid_loss/total_tokens_valid)
            train_loss.append(total_train_loss/total_tokens_train)

            print(f" Training loss = {train_loss[-1]} | Validation Loss = {valid_loss[-1]} | BPC = {bpc}")

            if best_valid_loss > valid_loss[-1]:
                os.makedirs(args.output_model_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_model_path,"model.pth"))
                best_valid_loss = valid_loss[-1]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
