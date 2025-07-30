import random
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 8 # Context length
SEQ_LEN = 2 # Equivalent to conv1d stride, i.e two chars embedding are concatenated

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def buil_vocab(fname):
    words = open(fname, 'r').read().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi["."] = 0
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos, words

def build_dataset(words, stoi):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return {"X":torch.tensor(X), "Y":torch.tensor(Y)}

class FlattenWavenet(nn.Module):
    """Implements custom Flatten wherer two consecutive chars embedding are concatenated
    """
    def __init__(self, seq_len):
        super(FlattenWavenet, self).__init__()
        self.seq_len = seq_len

    def forward(self, x):
        # x is [bs x context_length x embed_dim]
        bs, cl, dim = x.shape
        x = x.reshape(bs, cl // self.seq_len, dim * self.seq_len)
        if x.size(1) == 1:
            x = x.squeeze(1)
        return x

class BatchNormMod(nn.BatchNorm1d):

    def __init__(self,*args, **kargs):
        super(BatchNormMod, self).__init__(*args, **kargs)

    def forward(self, x):
        if x.ndim == 3:
            bs, seq_len, dim = x.shape
            x = x.permute(0, 2, 1)
            x = super(BatchNormMod, self).forward(x)
            return x.permute(0, 2, 1)
        elif x.ndim == 2:
            return super(BatchNormMod, self).forward(x)


def get_model(embed_dim, hidden_dim, out_dim, stoi):
    _model = nn.Sequential(
        nn.Embedding(len(stoi), embed_dim),

        FlattenWavenet(SEQ_LEN),
        nn.Linear(embed_dim * SEQ_LEN, hidden_dim, bias=False),
        BatchNormMod(hidden_dim),
        nn.ReLU(),

        FlattenWavenet(SEQ_LEN),
        nn.Linear(hidden_dim * SEQ_LEN, hidden_dim, bias=False),
        BatchNormMod(hidden_dim),
        nn.ReLU(),

        FlattenWavenet(SEQ_LEN),
        nn.Linear(hidden_dim * SEQ_LEN, hidden_dim, bias=False),
        BatchNormMod(hidden_dim),
        nn.ReLU(),

        nn.Linear(hidden_dim, len(stoi))
    )
    return _model

def train(model:nn.Module, train_dataset, val_dataset,  steps, bs, initlr):
    # Need to wrap in nn.Parameter, oherwise facing issue after moving to cuda
    losses = []
    i = 0
    model.to(device)
    model.train()
    for step in range(steps):
        X = train_dataset["X"][i * bs: i * bs + bs].to(device)
        Y = train_dataset["Y"][i * bs: i * bs + bs].to(device)
        if len(X) == 0:
            i = 0
            X = train_dataset["X"][i * bs: i * bs + bs].to(device)
            Y = train_dataset["Y"][i * bs: i * bs + bs].to(device)
        ypred = model(X)
        loss = F.cross_entropy(ypred, Y)
        for p in model.parameters():
            p.grad = None
        loss.backward()
        if step < 0.6 * steps:
            lr = initlr
        else:
            lr = initlr / 10
        for p in model.parameters():
            p.data += -lr * p.grad
        losses.append(loss)
        i += 1

        # val step
        if ((step + 1) % 100 == 0):
            model.eval()
            X = val_dataset["X"].to(device)
            Y = val_dataset["Y"].to(device)
            ypred = model(X)
            val_loss = F.cross_entropy(ypred, Y)
            print(f"[{step + 1}]/[{steps}]: train loss is {sum(losses)/ len(losses):.4f}: val loss is {val_loss:.4f}")

def get_opts(cargs=[]):
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--fname", default="names.txt")
    parser.add_argument("--embed_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)

    if cargs:
        opts = parser.parse_args(cargs)
    else:
        opts = parser.parse_args()
    return opts

def main(opts):
    stoi, itos, words = buil_vocab(opts.fname)
    random.shuffle(words)
    train_words = words[:int(0.8 * len(words))]
    val_words = words[len(train_words) : len(train_words) + int(0.1 * len(words))]
    test_words = words[len(train_words) + len(val_words): ]
    train_dataset = build_dataset(train_words, stoi)
    val_dataset = build_dataset(val_words, stoi)
    out_dim = len(stoi)
    model = get_model(opts.embed_dim, opts.hidden_dim, out_dim, stoi)
    train(
        model,
        train_dataset,
        val_dataset,
        opts.steps,
        opts.bs,
        opts.lr
    )

if __name__ == '__main__':
    cargs = []
    cargs.extend(["--steps", "20000"])
    opts = get_opts(cargs)
    main(opts)
