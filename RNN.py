import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data
file_name = 'divinacommedia.txt'
fulltext = open(file_name, 'rb').read().decode(encoding='utf-8').lower()
chars = sorted(list(set(fulltext)))
fulltext_len, vocab_size = len(fulltext), len(chars)
print('data has %d characters, %d unique.' % (fulltext_len, vocab_size))

# build the vocabulary of characters and mappings to / from integers
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_ix[c] for c in s]  # encoder : takes a string , outputs a list of integers
decode = lambda l: ''.join([ix_to_char[i] for i in l])  # decoder : takes a list of integers, outputs a string
encoded_text = encode(fulltext)
data_as_tensor = torch.tensor(encoded_text, dtype=torch.long)

# Hyper-parameters
input_size = vocab_size
output_size = vocab_size
seq_length = 25
step_size = 3  # shift of sequences

hidden_size = 200
batch_size = 128
num_layers = 8
num_epochs = 5
learning_rate = 0.01


# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
# batch_size = N, seq_length = L, D=1 (directions), Hin = input_size, Hout = hidden_size

def one_hot(tensor):  # one-hot encoding (1 of k representation)
    return (F.one_hot(tensor, num_classes=vocab_size)).to(torch.float32)


def index_from_hot(encoded_tensor):  # returns index of encoded character
    return torch.argmax(encoded_tensor, dim=0)


def char_from_hot(encoded_tensor):  # returns character
    return ix_to_char[torch.argmax(encoded_tensor, dim=0).item()]


def initialize_seq(train=True):  # initialize inputs and targets sequences
    print(f'initializing sequences...')
    k = int(0.8 * fulltext_len)
    if train:
        text = encoded_text[:k]  # train, 80%
    else:
        text = encoded_text[k:]  # validation, 20%

    inputs = []  # input sequences
    targets = []  # target characters for each input seq

    text_len = len(text)
    for i in range(0, text_len - seq_length, step_size):
        inputs.append(text[i: i + seq_length])
        targets.append(text[i + seq_length])

    input_tensor = torch.tensor(inputs)
    target_tensor = torch.tensor(targets)
    input_hot = one_hot(input_tensor)  # torch.Size([dim, L, Hin]). dim = # of sequences obtained
    target_hot = one_hot(target_tensor)  # torch.Size([dim, Hout]).
    num_seq = input_hot.size(0)  # dim

    # split input and target into batches of dimension batch_size
    num_batches = int(float(num_seq - batch_size) / float(batch_size))
    # we have num_batches input batches, each of these has length batch_size, and each element is a sequence
    # of length seq_length. each char of the sequence has dimension 37 (input_size)
    batched_input = torch.zeros(num_batches, batch_size, seq_length, input_size)
    # for the target we lack seq_length dimension since it's just a series of one char
    batched_target = torch.zeros(num_batches, batch_size, input_size)

    b = 0  # batch index
    seq_range = 0  # range of sequences to take from input in order to fill batch b
    while b < num_batches:
        s = 0  # index for batch element
        for seq in range(seq_range, seq_range + batch_size, 1):
            # Each batch element will be a sequence from input
            batched_input[b] = input_hot[seq]
            batched_target[b] = target_hot[seq]
            s += 1
        b += 1  # batch filled: go to new batch
        if seq_range < num_seq - batch_size:
            seq_range += batch_size  # but fill it with 100 new sequences (batch_size = 100)
        else:
            break
    return batched_input, batched_target


# initialize inputs and targets for train and validation
Xtr, Ytr = initialize_seq()  # training set
Xev, Yev = initialize_seq(train=False)  # validation set

layer_norm = nn.LayerNorm(vocab_size)  # layer normalization


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)  # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer

    def forward(self, x):
        # input shape: (N, L, Hin) or (L, Hin) for unbatched input
        out, hidden_state = self.rnn(x)
        # out is (batch_size, seq_length, hidden_size) or (seq_length, hidden_size) for unbatched

        # out: (N, L, Hout). But we don't need all the chars of the sequence, just the last one
        if x.size(0) > 1:
            out = out[:, -1, :]  # select last char: from (N, L, Hout) --> output dim: (N, Hout)

        out = self.fc(out)  # logits
        return out, hidden_state

    def sample(self, seed):  # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            if len(seed) <= 1:  # if seed is a single char
                seed = one_hot(torch.tensor(char_to_ix[seed]))
                seed = torch.unsqueeze(seed, 0)  # add dummy dimension for matching size
            else:
                seed = one_hot(torch.tensor(encode(seed)))
            output, _ = self.rnn(seed)
            output = output[-1, :]  # select last char probabilities
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            return ix_to_char[sample_ix]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer

    def forward(self, x):
        # input shape: (N, L, Hin) or (L, Hin) for unbatched input
        out, hidden_state = self.lstm(x)
        # out is (batch_size, seq_length, hidden_size) or (seq_length, hidden_size) for unbatched

        # out: (N, L, Hout). But we don't need all the chars of the sequence, just the last one
        if x.size(0) > 1:
            out = out[:, -1, :]  # select last char: from (N, L, Hout) --> output dim: (N, Hout)

        out = self.fc(out)  # logits
        return out, hidden_state

    def sample(self, seed):  # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            if len(seed) <= 1:  # if seed is a single char
                seed = one_hot(torch.tensor(char_to_ix[seed]))
                seed = torch.unsqueeze(seed, 0)  # add dummy dimension for matching size
            else:
                seed = one_hot(torch.tensor(encode(seed)))
            output, _ = self.lstm(seed)
            output = output[-1, :]  # select last char probabilities
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            return ix_to_char[sample_ix]


# model
model = LSTM(input_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
n_iter = 10000
current_loss_tr = 0
current_loss_ev = 0
tr_losses = []
ev_losses = []
plot_steps = n_iter / 100
sample_steps = n_iter / 100

tr_batches = Xtr.size(0)
ev_batches = Xev.size(0)

print('starting training and evaluation...')
for epoch in range(num_epochs):
    for i in range(n_iter):
        # training
        # select random batch:
        b_tr = random.randint(0, tr_batches - 1)
        b_ev = random.randint(0, ev_batches - 1)

        Xb = layer_norm(Xtr[b_tr])
        Yb = Ytr[b_tr]

        # forward pass
        model.train()
        outputs, h_n = model(Xb)
        tr_loss = criterion(outputs, Yb)

        # backward and optimize
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        current_loss_tr += tr_loss.item()

        # evaluation
        model.eval()
        with torch.no_grad():
            # evaluate loss on eval set
            Xevb = layer_norm(Xev[b_ev])
            Yevb = Yev[b_ev]
            out, _ = model(Xevb)
            ev_loss = criterion(out, Yevb)
            current_loss_ev += ev_loss.item()

        if (i + 1) % plot_steps == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_iter}], train Loss: {tr_loss.item():.4f}, eval Loss: {ev_loss.item():.4f}')

            tr_losses.append(current_loss_tr / plot_steps)
            ev_losses.append(current_loss_ev / plot_steps)
            current_loss_tr = 0
            current_loss_ev = 0

    plt.figure()
    plt.plot(tr_losses, color='blue', label='training loss')
    plt.plot(ev_losses, color='orange', label='evaluation loss')
    plt.title(f'Training loss vs evaluation loss, epoch {epoch}')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

# sampling
seed_seq = 'nel mezzo del cammin di nostra vita'
sample_seq = [c for c in seed_seq]
sample_len = 75
for k in range(sample_len):
    prediction = model.sample(sample_seq)
    sample_seq.append(prediction)
txt = ''.join(sample_seq)
print(f'sampled text: {txt}')


