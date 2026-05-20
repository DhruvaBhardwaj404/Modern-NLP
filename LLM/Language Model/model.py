import torch
import torch.nn as nn
from typing import Any, Dict, List
from torch.nn.utils.rnn import pad_sequence

MAX_LEN = 512
torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingLayer(nn.Module, ):
    def __init__(self, vocab, d_model, device):
        super().__init__()
        self.emb = nn.Embedding(vocab,d_model).to(device=device)

    def forward(self, input_ids: tuple):
        return self.emb(input_ids)

class PosEncoding(nn.Module):
    def __init__(self,d_model,max_len, dev):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float, device= dev).unsqueeze(1)
        i = torch.arange(0,d_model//2, dtype=torch.float, device= dev).unsqueeze(0)

        denom = torch.pow(10000.0, (2.0 * i / d_model)).to(torch.float32)
        ang = torch.divide(pos,denom)

        pe = torch.zeros(max_len,d_model, device=dev)
        pe[:,0::2] =  torch.sin(ang)
        pe[:,1::2] = torch.cos(ang)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_emb:torch.tensor):
        seq_len = input_emb.shape[1]
        output_emb = input_emb + self.pe[:,:seq_len,:]
        return output_emb
        # return input_emb.add_(self.pe[:, :seq_len, :])


class RotPosEmbedding(nn.Module):
    def __init__(self,d_head,max_len, dev):
        super().__init__()
        self.d_head = d_head
        pos = torch.arange(0, max_len, dtype=torch.float, device=dev).unsqueeze(1)
        i = torch.arange(0, d_head// 2, dtype=torch.float, device=dev).unsqueeze(0)
        theta = 1.0/torch.pow(10000.0,(2*i)/d_head)
        ang_half = pos@theta
        ang = ang_half.repeat_interleave(2, dim=-1)

        sin = torch.sin(ang).unsqueeze(0)
        cos = torch.cos(ang).unsqueeze(0)

        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)

    def forward(self, input_emb:torch.tensor):
        batch, seq_len, _ = input_emb.shape
        input_emb_rot = torch.stack([- input_emb[:, :, 1::2], input_emb[:, :, 0::2]], dim=-1).flatten(-2)

        if seq_len <= MAX_LEN:
            output_emb = input_emb*self.cos[:,:seq_len,:] + input_emb_rot*self.sin[:,:seq_len,:]
        else:
            ext_len = seq_len - self.sin.shape[1]
            sin = torch.cat([self.sin, self.sin[:, -1:, :].expand(-1, ext_len, -1)], dim=1)
            cos = torch.cat([self.cos, self.cos[:, -1:, :].expand(-1, ext_len, -1)], dim=1)
            output_emb = input_emb * cos + input_emb_rot * sin
        return output_emb

# class AttentionCustomLayer(nn.Module):
#     def __init__(self):


class TransformerCustomLayer(nn.Module):
    def __init__(self, d_model, d_head, n_heads, mode="standard",tau=None,device="cpu",rope=None):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.rope = rope
        self.device = device
        self.tau = torch.tensor(tau, dtype=torch.float32, device=device) if tau is not None else None
        self.emb_d = d_head*n_heads
        # self.W_K = nn.Linear(self.d_model,self.emb_d,bias=False)
        # self.W_V = nn.Linear(self.d_model,self.emb_d,bias=False)
        # self.W_Q = nn.Linear(self.d_model,self.emb_d,bias=False)

        self.W_KQV = nn.Linear(self.d_model,3*self.emb_d,bias = False).to(device = self.device)
        #self.LayerNorm1 = nn.LayerNorm(self.d_model,elementwise_affine=True).to(device = self.device)
        self.LayerNorm1 = nn.RMSNorm(self.d_model).to(device=self.device)
        self.W_O =  nn.Linear(self.d_model,self.d_model,bias=False).to(device = self.device)
        #self.LayerNorm2 = nn.LayerNorm(self.d_model, elementwise_affine=True).to(device = self.device)
        self.LayerNorm2 = nn.RMSNorm(self.d_model).to(device=self.device)
        self.W_up = nn.Linear(self.d_model,4*self.d_model).to(device = self.device)
        self.W_down = nn.Linear(4*self.d_model,self.d_model).to(device = self.device)
        self.W_sgate = nn.Linear(self.d_model, 4 * self.d_model, bias=False).to(device=self.device)
        mask = torch.triu(torch.ones(MAX_LEN, MAX_LEN, device=self.device), diagonal=1).bool().to(device = self.device)
        self.register_buffer('mask',mask)
        self.a_dropout = nn.Dropout(0.2)
        self.f_dropout = nn.Dropout(0.1)
        self.scale_inv = (1.0/(torch.sqrt(torch.tensor(self.d_head,dtype=torch.float32)))).to(device = self.device)

    def forward(self, input_emb, attention_mask):
        # input_emb, attention_mask = input_tuple
        batch, seq, _ = input_emb.shape

        x = self.LayerNorm1(input_emb)

        # K = self.W_K(x).reshape(x.shape[0], x.shape[1], self.n_heads,self.d_head).transpose(1, 2)
        # V = self.W_V(x).reshape(x.shape[0], x.shape[1], self.n_heads,self.d_head).transpose(1, 2)
        # Q = self.W_Q(x).reshape(x.shape[0], x.shape[1], self.n_heads,self.d_head).transpose(1, 2)
        KQV = self.W_KQV(x)
        # K, Q, V = KQV.split(self.emb_d, dim=-1)
        #
        # K = K.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        # Q = Q.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        # V = V.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        KQV = KQV.view(batch, seq, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        K, Q, V = KQV[0], KQV[1], KQV[2]

        # S = (Q@K.transpose(-2, -1))*self.scale_inv

        Q = Q.reshape(batch * self.n_heads, seq, self.d_head)
        K = K.reshape(batch * self.n_heads, seq, self.d_head)

        K_rope = self.rope(K)
        Q_rope = self.rope(Q)

        K_rope = K_rope.transpose(-2, -1)

        V_flat = V.reshape(batch * self.n_heads, seq, self.d_head)

        if self.mode == 'standard':
            S = torch.baddbmm(attention_mask, Q_rope, K_rope, beta=1.0, alpha=self.scale_inv)

        else:
            S = torch.bmm(Q_rope,K_rope).mul_(self.scale_inv)
            S = self.tau * torch.tanh(S)
            S = S.add_(attention_mask)

        # S = S.masked_fill(self.mask[:seq,:seq], float('-inf'))

        # pad_mask = (attention_mask == 0).view(batch, 1, 1, seq)
        # mask = pad_mask | self.mask[:seq,:seq].unsqueeze(0).unsqueeze(0)
        # S = S.masked_fill_(mask, -1e9)


        Attn = torch.softmax(S, axis = -1)
        Attn = self.a_dropout(Attn)
        # Attn = torch.nan_to_num(Attn, nan=0.0)
        heads = torch.matmul(Attn,V_flat)#.transpose(1, 2).reshape(batch, seq, self.emb_d)
        heads = heads.view(batch, self.n_heads, seq, self.d_head)
        heads = heads.transpose(1, 2).reshape(batch, seq, self.emb_d)
        x = self.W_O(heads) + input_emb

        x_norm = self.LayerNorm2(x)
        # x_up = self.W_up(x_norm)
        # x_gelu = nn.functional.gelu(x_up)
        # output = self.f_dropout(self.W_down(x_gelu) + x)
        # output = x + self.W_down(nn.functional.gelu(self.W_up(self.LayerNorm2(x))))

        x_gate = nn.functional.silu(self.W_sgate(x_norm))
        x_up = self.W_up(x_norm)
        output = self.f_dropout(self.W_down(x_gate * x_up) + x)

        return output


class LanguageModel(nn.Module):

    def __init__(self, config: Dict[str, Any],device = None):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        # d_model, n_heads,d_head,n_layers, vocab_size,
        # mode, tau

        super().__init__()

        self.device = device if device is not None else DEVICE

        self.vocab_size = self.config["vocab_size"]
        self.mode = self.config["mode"]
        if self.mode != "standard":
            self.tau = self.config["tau"]
        else:
            self.tau = None
        self.d_head = self.config["d_head"]
        self.d_model = self.config["d_model"]
        self.n_heads = self.config["n_heads"]
        self.n_layers = self.config["n_layers"]
        self.max_len = MAX_LEN
        self.emb_layer = EmbeddingLayer(self.vocab_size, self.d_model,self.device)
        #self.pos_emb_layer = PosEncoding(self.d_model,self.max_len,dev=self.device)
        self.rope = RotPosEmbedding(self.d_head,self.max_len,dev=self.device)
        self.decoder = nn.ModuleList(
                                   [TransformerCustomLayer(self.d_model,self.d_head,self.n_heads,self.mode,self.tau,self.device,self.rope) for _ in range(self.n_layers)],
                                   )
        #self.layer_norm_final = nn.LayerNorm(self.d_model).to(device = self.device)
        self.layer_norm_final = nn.RMSNorm(self.d_model).to(device = self.device)
        self.logits = nn.Linear(self.d_model, self.vocab_size, bias=False).to(device = self.device)
        self.logits.weight = self.emb_layer.emb.weight

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """

        self.emb_layer.emb.weight.data.copy_(weights["W_vocab"].t().detach().to(device=self.device, dtype = torch.float32))

        for n in range(self.n_layers):
            K_keys = [f"W_{n+1}_K_{i+1}" for i in range(self.n_heads)]
            V_keys = [f"W_{n+1}_V_{i+1}" for i in range(self.n_heads)]
            Q_keys = [f"W_{n+1}_Q_{i+1}" for i in range(self.n_heads)]

            k_heads = []
            for k in K_keys:
                w = weights[k].detach().to(torch.float32)
                k_heads.append(w)

            W_K_temp = torch.cat(k_heads, dim=0)

            # self.decoder[n].W_K.weight.data.copy_(W_K_temp)

            q_heads = []
            for q in Q_keys:
                w = weights[q].detach().to(torch.float32)
                q_heads.append(w)

            W_Q_temp = torch.cat(q_heads, dim=0)

            # self.decoder[n].W_Q.weight.data.copy_(W_Q_temp)

            v_heads = []
            for v in V_keys:
                w = weights[v].detach().to(torch.float32)
                v_heads.append(w)

            W_V_temp = torch.cat(v_heads, dim=0)
            W_KQV_temp = torch.cat([W_K_temp,W_Q_temp,W_V_temp],dim=1)
            # self.decoder[n].W_V.weight.data.copy_(W_V_temp)
            self.decoder[n].W_KQV.weight.data.copy_(W_KQV_temp.t().to(device=self.device,dtype = torch.float32))
            self.decoder[n].W_O.weight.data.copy_(weights[f"W_{n+1}_O"].t().detach().to(device=self.device,dtype = torch.float32))

            self.decoder[n].W_up.bias.data.copy_(weights[f"b_{n+1}_up"].detach().to(device=self.device,dtype = torch.float32))
            self.decoder[n].W_up.weight.data.copy_(weights[f"W_{n+1}_up"].t().detach().to(device=self.device,dtype = torch.float32))

            self.decoder[n].W_down.bias.data.copy_(weights[f"b_{n+1}_down"].detach().to(device=self.device,dtype = torch.float32))
            self.decoder[n].W_down.weight.data.copy_(weights[f"W_{n+1}_down"].t().detach().to(device=self.device,dtype = torch.float32))

            self.decoder[n].LayerNorm1.bias.data.copy_(weights[f"beta_{n+1}_1"].detach().to(device=self.device,dtype = torch.float32))
            self.decoder[n].LayerNorm1.weight.data.copy_(weights[f"gamma_{n+1}_1"].detach().to(device=self.device,dtype = torch.float32))

            self.decoder[n].LayerNorm2.bias.data.copy_(weights[f"beta_{n+1}_2"].detach().to(device=self.device,dtype = torch.float32))
            self.decoder[n].LayerNorm2.weight.data.copy_(weights[f"gamma_{n+1}_2"].detach().to(device=self.device,dtype = torch.float32))

        self.layer_norm_final.bias.data.copy_(weights[f"beta_final"].detach().to(device=self.device,dtype = torch.float32))
        self.layer_norm_final.weight.data.copy_(weights[f"gamma_final"].detach().to(device=self.device,dtype = torch.float32))

        self.logits.weight.data.copy_(weights["W_devocab"].t().detach().to(device=self.device,dtype = torch.float32))


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        x_emb = self.emb_layer(input_ids)
        #x_emb = self.pos_emb_layer(x_emb)

        batch, seq = input_ids.shape
        pad_mask = (attention_mask == 0).view(batch, 1, 1, seq)

        if seq <= MAX_LEN:
            causal_mask = self.decoder[0].mask[:seq, :seq].view(1, 1, seq, seq)
        else:
            causal_mask = torch.triu(torch.ones(seq, seq, device=x_emb.device), diagonal=1).bool().view(1, 1, seq, seq)

        S_mask = (pad_mask | causal_mask).to(x_emb.dtype).mul(-1e9)
        S_mask = S_mask.expand(batch, self.n_heads, seq, seq).reshape(-1, seq, seq)
        # decoder_out, mask = self.decoder((x_emb_pos,attention_mask))
        for layer in self.decoder:
            x_emb = layer(x_emb, S_mask)

        x_norm = self.layer_norm_final(x_emb)
        logits = self.logits(x_norm)

        return logits



def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    model = LanguageModel(config)
    model.set_weights(weights)
    return model


def collate_fn(batch):

    if isinstance(batch,list):
        batch = {key: [d[key] for d in batch] for key in batch[0]}
        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=0)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=0)
        labels = pad_sequence(batch["labels"], batch_first=True, padding_value=0)
        return input_ids, attention_mask, labels
    else:

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=0)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=0)
        # labels = pad_sequence(batch["label"], batch_first=True, padding_value=0)
        return {"input_ids":input_ids, "attention_mask": attention_mask}


