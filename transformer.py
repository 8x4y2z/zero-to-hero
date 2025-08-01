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

def get_train_data(train_data, bs):
    while True:
        train_idx = torch.randint(len(train_data) - BLOCK_SIZE, (bs,))
        train_x = torch.stack([train_data[i:i+BLOCK_SIZE] for i in train_idx])
        train_y = torch.stack([train_data[i+1:i+1+BLOCK_SIZE] for i in train_idx])
        yield {"X":train_x, "Y":train_y}

def get_val_data(val_data):
    val_x = torch.stack([val_data[i:i+BLOCK_SIZE] for i in range(len(val_data) - BLOCK_SIZE)])
    val_y = torch.stack([val_data[i+1:i+1+BLOCK_SIZE] for i in range(len(val_data) - BLOCK_SIZE)])
    return {"X":val_x, "Y":val_y}

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, nheads, ff_dim, block_size):
        super(MultiHeadAttentionLayer, self).__init__()
        self.head_dim = hidden_dim // nheads
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.query = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.value = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim))
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size )))

    def forward(self, x):
        # Following is self attention as keys values and queries all come from x
        # In cross attention only queries come from x, where as keys and values are from a different
        # vector.

        b, t, c = x.shape
        q = self.query(x) # Every token produces a query vector. Query roughly means what I'm looking for
        k = self.key(x) # Every token also produces a key vecto. Key roughtly means what I contain
        v = self.value(x) # Value is just for aggregation at the final step

        q = q.view(b, t, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, t, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, t, self.nheads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.transpose(-2, -1)) / self.scale # Scale is needed to stablize the upcoming softmax, so that it won't saturate,
        # and variance is reduced, specially at initialization
        energy = energy.masked_fill(self.mask[:t, :t] == 0 , float("-inf"))
        attention = F.softmax(energy, -1)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(b, t, self.hidden_dim)
        out = self.ffn(out)
        return out

class Transformer(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, nheads,
                 ffn_dim,
                 block_size):
        super(Transformer, self).__init__()
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.self_attention = MultiHeadAttentionLayer(embed_dim, hidden_dim,
                                                      nheads,
                                                      ffn_dim,
                                                      block_size)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        b, t = x.shape
        inp_embed = self.input_embedding(x)
        tok_embed = self.position_embedding(torch.arange(t, device=device))
        x = inp_embed + tok_embed
        x = self.self_attention(x)
        out = self.output_head(x)
        return out

def train(model:nn.Module, train_dataset, val_dataset, steps, bs, optim):
    model.to(device)
    model.train()
    for step in range(steps):
        batched_inps = next(train_dataset)
        X = batched_inps["X"].to(device)
        Y = batched_inps["Y"].to(device)
        ypred = model(X)
        bs, ts, d = ypred.shape
        loss = F.cross_entropy(ypred.view(bs * ts, d), Y.view(bs * ts))
        model.zero_grad()
        loss.backward()
        optim.step()

        # val step
        if ((step + 1) % 100 == 0):
            val_loss = []
            i = 0
            vbs = 10 * bs
            with torch.no_grad():
                model.eval()
                X = val_dataset["X"]
                Y = val_dataset["Y"]
                while True:
                    xb = X[i * vbs : i * vbs + vbs].to(device)
                    yb = Y[i * vbs : i * vbs + vbs].to(device)
                    if len(xb) == 0:
                        break
                    ypred = model(xb)
                    sbs, ts, d = ypred.shape
                    vloss = F.cross_entropy(ypred.view(sbs * ts, d), yb.view(sbs * ts))
                    val_loss.append(vloss)
                    i += 1
                print(f"[{step + 1}]/[{steps}]: train loss is {loss.item():.4f}: val loss is {sum(val_loss) / len(val_loss):.4f}")

def generate(model, idx:torch.Tensor, max_length, block_size):
    model.to(device)
    model.eval()
    idx = idx.to(device)
    for _ in range(max_length):
        logits = model(idx[:, -block_size:]) # [bs x ts x C]
        logits = logits[:, -1, :] # [bs x C]
        probs = F.softmax(logits, dim=-1)
        _next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, _next], dim=1)
    return idx
            

def get_opts(cargs=[]):
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--fname", default="input.txt")
    parser.add_argument("--embed_dim", default=50, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--max_length", default=500, type=int)
    parser.add_argument("--nlayers", default=6, type=int)
    parser.add_argument("--ffn_dim", default=256, type=int)
    
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
    model = Transformer(vocab_size,
                        opts.embed_dim,
                        opts.hidden_dim,
                        opts.nheads,
                        opts.ffn_dim,
                        BLOCK_SIZE)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = get_train_data(train_data, opts.bs)
    val_dataset = get_val_data(val_data)
    optim = torch.optim.AdamW(model.parameters(),lr=opts.lr)
    train(
        model,
        train_dataset,
        val_dataset,
        opts.steps,
        opts.bs,
        optim
    )
    out = generate(model, torch.zeros((1,1),dtype=torch.long), opts.max_length, BLOCK_SIZE)
    print(decode(out[0].tolist()))

if __name__ == '__main__':
    cargs = []
    cargs.extend(["--steps", "1000"])
    cargs.extend(["--lr", "1e-3"])
    opts = get_opts(cargs)
    main(opts)
