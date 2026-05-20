import os.path
import torch
from partb.bpe_tokenizer import BPETokenizer
from parta.model import *
import wandb
import optuna
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import warnings
warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5

class CustomDataset(Dataset):
    def __init__(self, x):
        self.data = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x_tensor = torch.tensor(self.data[item][:-1])
        y_tensor = torch.tensor(self.data[item][1:])
        return {
            "input_ids": x_tensor,
            "attention_mask": torch.ones_like(x_tensor),
            "labels": y_tensor
        }

def create_dataset(ttext):
    x = []

    torch.manual_seed(50)
    char_count = 0
    for line in ttext:
        if torch.randint(0,10,[1]) .item() <= 3:
            x.append(torch.tensor(line))
            char_count+=len(line)
    return CustomDataset(x), char_count

def objective(trial,train_dataloader,valid_dataloader, total_valid_chars, vocab_size):
    d_heads = trial.suggest_categorical("d_heads", [4, 8, 16])
    n_heads = trial.suggest_categorical("n_heads", [4, 8, 16, 32])
    d_model = d_heads*n_heads
    n_layers = trial.suggest_categorical("n_layers", [4, 8, 16, 32])
    mode = trial.suggest_categorical("mode", ["standard","tanh_clipped"])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)


    config = {
        "d_model": d_model, "n_heads": n_heads, "d_head": d_heads,
        "n_layers": n_layers, "vocab_size": vocab_size,
        "mode": mode, "tau": 1.5
    }

    run = wandb.init(
        project="A2-NLP",
        name=f"trial_{trial.number}",
        config={**trial.params, "d_model": d_model},
        reinit=True,
        group="optuna_search"
    )

    try:
        model = LanguageModel(config, device=DEVICE)
        train_loss = []
        valid_loss = []
        bpc_all = []
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        warmup_steps = int(len(train_dataloader)*EPOCHS*0.1)
        t_max = int(len(train_dataloader)*EPOCHS) - warmup_steps
        linear_sch = LinearLR(optim, start_factor=0.1, total_iters=warmup_steps)
        cosine_sch = CosineAnnealingLR(optim, T_max=t_max)
        scheduler = SequentialLR(optim, schedulers=[linear_sch, cosine_sch], milestones=[warmup_steps])

        for e in range(EPOCHS):
            print(f"Epoch {e + 1}/{EPOCHS}==>")
            total_train_loss = 0.0
            model.train()
            total_tokens_train = 0
            for x, attn_mask, y in train_dataloader:
                x, attn_mask, y = x.to(DEVICE), attn_mask.to(DEVICE), y.to(DEVICE)
                optim.zero_grad()
                logits = model(x, attn_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                with torch.no_grad():
                    total_train_loss += loss.item()
                    total_tokens_train += y.numel()
                scheduler.step()


            with torch.no_grad():
                total_valid_loss = 0.0
                total_tokens_val = 0
                model.eval()
                for x, attn_mask, y in valid_dataloader:
                    x, attn_mask, y = x.to(DEVICE), attn_mask.to(DEVICE), y.to(DEVICE)
                    logits = model(x, attn_mask)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    total_valid_loss += loss.item()
                    total_tokens_val += y.numel()

                bpc = total_valid_loss / (total_valid_chars * torch.log(torch.tensor(2)).item())
                bpc_all.append(bpc)
                valid_loss.append(total_valid_loss / total_tokens_val)
                train_loss.append(total_train_loss / total_tokens_train)
                print(f" Training loss = {train_loss[-1]} | Validation Loss = {valid_loss[-1]} | BPC = {bpc}")
                wandb.log({"epoch": e, "train_loss": train_loss[-1], "val_bpc": bpc, "validation_loss":valid_loss[-1]})

                # if best_valid_loss > valid_loss[-1]:
                #     torch.save(model.state_dict(), os.path.join(args.output_model_path, "model.pth"))

                trial.report(bpc, e)

                if trial.should_prune():
                    run.finish()
                    raise optuna.exceptions.TrialPruned()

        run.finish()
        return bpc

    except Exception as e:
        print(f"Failed => {str(e)}")
        run.finish()
        return float('inf')

def main(args):
    with torch.no_grad():
        tokenizer: BPETokenizer = BPETokenizer()
        tokenizer.load(args.tokenizer_path)
        train_ttext = []
        with open(args.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_ttext.append(tokenizer.encode(line))

        valid_ttext = []

        with open(args.valid_path, 'r', encoding='utf-8') as f:
            for line in f:
                valid_ttext.append(tokenizer.encode(line))

        train_dataset, train_char_count = create_dataset(train_ttext)
        valid_dataset, valid_char_count = create_dataset(valid_ttext)
        train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    study_name = "A2-NLP"
    storage_name = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner
    )

    study.optimize(
        lambda trial: objective(trial, train_dataloader, valid_dataloader, valid_char_count, tokenizer.get_vocab_size()),
        n_trials=20
    )

    print(f"Best BPC: {study.best_value}")
    print(f"Best Params: {study.best_params}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
