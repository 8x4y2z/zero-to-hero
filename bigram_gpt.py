from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BLOCK_SIZE = 8

def buil_vocab(fname):
    text = open(fname, 'r').read()
    chars = sorted(list(set(text)))
    stoi = {s:i for i,s in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos, text

def build_datasets(text, encode):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    train_x = torch.stack([train_data[i:i+BLOCK_SIZE] for i in range(len(train_data) - BLOCK_SIZE)])
    train_y = torch.stack([train_data[i+1:i+1+BLOCK_SIZE] for i in range(len(train_data) - BLOCK_SIZE)])

    val_x = torch.stack([val_data[i:i+BLOCK_SIZE] for i in range(len(val_data) - BLOCK_SIZE)])
    val_y = torch.stack([val_data[i+1:i+1+BLOCK_SIZE] for i in range(len(val_data) - BLOCK_SIZE)])

    return {"X":train_x, "Y":train_y}, {"X":val_x, "Y":val_y}

class Bigram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Bigram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        # x:[b]
        y = self.embedding(x) # [b x embed_dim]
        return y

def train(model:nn.Module, train_dataset, val_dataset, steps, bs, optim):
    model.to(device)
    model.train()
    i = 0
    for step in range(steps):
        X = train_dataset["X"][i * bs: i * bs + bs].to(device)
        Y = train_dataset["Y"][i * bs: i * bs + bs].to(device)
        if len(X) == 0:
            i = 0
            X = train_dataset["X"][i * bs: i * bs + bs].to(device)
            Y = train_dataset["Y"][i * bs: i * bs + bs].to(device)
        ypred = model(X)
        bs, ts, d = ypred.shape
        loss = F.cross_entropy(ypred.view(bs * ts, d), Y.view(bs * ts))
        model.zero_grad()
        loss.backward()
        optim.step()

        # val step
        if ((step + 1) % 100 == 0):
            model.eval()
            X = val_dataset["X"].to(device)
            Y = val_dataset["Y"].to(device)
            ypred = model(X)
            bs, ts, d = ypred.shape
            val_loss = F.cross_entropy(ypred.view(bs * ts, d), Y.view(bs * ts))
            print(f"[{step + 1}]/[{steps}]: train loss is {loss.item():.4f}: val loss is {val_loss:.4f}")

def generate(model, idx:torch.Tensor, max_length):
    model.eval()
    for _ in range(max_length):
        logits = model(idx) # [bs x ts x C]
        logits = logits[:, -1, :] # [bs x C]
        probs = F.softmax(logits, dim=-1)
        _next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, _next], dim=1)
    return idx

def get_opts(cargs=[]):
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--fname", default="input.txt")
    parser.add_argument("--embed_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)

    if cargs:
        opts = parser.parse_args(cargs)
    else:
        opts = parser.parse_args()
    return opts

def main(opts):
    stoi, itos, text = buil_vocab(opts.fname)
    encode = lambda x: [stoi[xx] for xx in x]
    decode = lambda l: "".join([itos[xx] for xx in l])
    vocab_size = len(stoi)
    model = Bigram(vocab_size, vocab_size)
    train_dataset, val_dataset = build_datasets(text, encode)
    optim = torch.optim.AdamW(model.parameters(),lr=opts.lr)
    train(
        model,
        train_dataset,
        val_dataset,
        opts.steps,
        opts.bs,
        optim
    )
    out = generate(model, torch.zeros((1,1),dtype=torch.long), max_length=500)
    print(decode(out[0].tolist()))

if __name__ == '__main__':
    cargs = []
    cargs.extend(["--steps", "20000"])
    opts = get_opts(cargs)
    main(opts)
