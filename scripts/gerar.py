import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Arquitetura (igual à do notebook) ---
class Head(nn.Module):
    def __init__(self, d, head_dim, T):
        super().__init__()
        self.key = nn.Linear(d, head_dim, bias=False)
        self.query = nn.Linear(d, head_dim, bias=False)
        self.value = nn.Linear(d, head_dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))
        self.head_dim = head_dim

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        Tc = x.size(1)
        scores = scores.masked_fill(self.mask[:Tc, :Tc] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, T):
        super().__init__()
        self.heads = nn.ModuleList([Head(d, d // h, T) for _ in range(h)])
        self.projection = nn.Linear(d, d)

    def forward(self, x):
        return self.projection(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d, h, T):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attention = MultiHeadAttention(d, h, T)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d, T, L, h):
        super().__init__()
        self.T = T
        self.token_embedding = nn.Embedding(vocab_size, d)
        self.position_embedding = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, h, T) for _ in range(L)])
        self.ln_final = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, idx):
        B, Tc = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(Tc, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_final(x))


@torch.no_grad()
def generate(model, idx, n, temperature, top_k):
    for _ in range(n):
        logits = model(idx[:, -model.T:])[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def main():
    ap = argparse.ArgumentParser(description="Mini-Copilot: gera código com o MiniGPT treinado")
    ap.add_argument("--modelo", default="modelos/minigpt_v3.pt")
    ap.add_argument("--prompt", default=None, help="Se omitido, abre o modo interativo")
    ap.add_argument("--n", type=int, default=400, help="Caracteres a gerar")
    ap.add_argument("--temperatura", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.modelo, map_location="cpu")
    chars, hp = ckpt["chars"], ckpt["hparams"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    model = MiniGPT(len(chars), hp["d"], hp["T"], hp["L"], hp["h"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Modelo carregado: {args.modelo} (contexto {hp['T']}, vocabulário {len(chars)})")

    def gerar(prompt):
        prompt = prompt.replace("\\n", "\n")   # permite digitar \n como quebra de linha
        invalidos = sorted({c for c in prompt if c not in stoi})
        if invalidos:
            print("Caracteres fora do vocabulário:", invalidos)
            return
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
        out = generate(model, idx, args.n, args.temperatura, args.top_k)
        texto = "".join(itos[i] for i in out[0].tolist())
        print("-" * 60)
        print(texto.split("@@@ FIM @@@")[0])
        print("-" * 60)

    if args.prompt:
        gerar(args.prompt)
    else:
        print("Digite um começo de código (use \\n para quebra de linha). 'sair' encerra.")
        while True:
            p = input("\nprompt> ")
            if p.strip().lower() == "sair":
                break
            if p:
                gerar(p)


if __name__ == "__main__":
    main()