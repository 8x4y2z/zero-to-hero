from argparse import ArgumentParser
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 3
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

def model(embed_dim, hidden_dim, out_dim, stoi):
    embed = torch.randn(len(stoi), embed_dim)
    W1 = torch.randn(BLOCK_SIZE * embed_dim, hidden_dim)
    b1 = torch.rand(hidden_dim)

    W2 = torch.randn(hidden_dim, out_dim)
    b2 = torch.rand(out_dim)
    params = [embed, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    return params

def train(model_params, train_dataset, val_dataset,  epochs, bs, initlr):
    # Need to wrap in nn.Parameter, oherwise facing issue after moving to cuda
    model_params = [nn.Parameter(p.to(device)) for p in model_params]
    for epoch in range(epochs):
        step = 0
        losses = []
        val_loss = []
        while True:
            X = train_dataset["X"][step * bs: step * bs + bs].to(device)
            Y = train_dataset["Y"][step * bs: step * bs + bs].to(device)
            if len(X) == 0:
                break
            embed, w1, b1, w2, b2 = model_params
            Xemb = embed[X].float()
            h1 = F.relu(Xemb.view(len(X),-1 ) @ w1 + b1)
            ypred = h1 @ w2 + b2
            loss = F.cross_entropy(ypred, Y)
            for p in model_params:
                p.grad = None
            loss.backward()
            if epoch < 0.6 * epochs:
                lr = initlr
            else:
                lr = initlr / 10
            for p in model_params:
                p.data += -lr * p.grad
            losses.append(loss)
            step += 1
        step = 0
        while True:
            vbs = bs * 2
            X = val_dataset["X"][step * vbs: step * vbs + vbs].to(device)
            Y = val_dataset["Y"][step * vbs: step * vbs + vbs].to(device)
            if len(X) == 0:
                break
            embed, w1, b1, w2, b2 = model_params
            Xemb = embed[X]
            h1 = F.relu(Xemb.view(len(X),-1 ) @ w1 + b1)
            ypred = h1 @ w2 + b2
            loss = F.cross_entropy(ypred, Y)
            val_loss.append(loss)
            step += 1


        print(f"[{epoch + 1}]/[{epochs}]: train loss is {sum(losses)/ len(losses):.4f}: val loss is {sum(val_loss)/len(val_loss):.4f}")
        shuffled = list(train_dataset.items())
        random.shuffle(shuffled)
        train_dataset = dict(shuffled)

def main(opts):
    stoi, itos, words = buil_vocab(opts.fname)
    random.shuffle(words)
    train_words = words[:int(0.8 * len(words))]
    val_words = words[len(train_words) : len(train_words) + int(0.1 * len(words))]
    test_words = words[len(train_words) + len(val_words): ]
    train_dataset = build_dataset(train_words, stoi)
    val_dataset = build_dataset(val_words, stoi)
    out_dim = len(stoi)
    model_params = model(opts.embed_dim, opts.hidden_dim, out_dim, stoi)
    train(
        model_params,
        train_dataset,
        val_dataset,
        opts.epochs,
        opts.bs,
        opts.lr
    )

def get_opts(cargs=[]):
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--fname", default="names.txt")
    parser.add_argument("--embed_dim", default=10, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)

    if cargs:
        opts = parser.parse_args(cargs)
    else:
        opts = parser.parse_args()
    return opts
if __name__ == '__main__':
    cargs = []
    opts = get_opts(cargs)
    main(opts)
