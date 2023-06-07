import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

start = time.time()

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data
file_name = 'divinacommedia.txt'
fulltext = open(file_name, 'rb').read().decode(encoding='utf-8').lower()
chars = sorted(list(set(fulltext)))
# print(chars)
fulltext_len, vocab_size = len(fulltext), len(chars)
print('data has %d characters, %d unique.' % (fulltext_len, vocab_size))

# build the vocabulary of characters and mappings to / from integers
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_ix[c] for c in s]  # encoder : take a string , output a list of integers
decode = lambda l: ''.join([ix_to_char[i] for i in l])  # decoder : take a list of integers, output a string
encoded_text = encode(fulltext)
data_as_tensor = torch.tensor(encoded_text, dtype=torch.long)
# split training and evaluation data later on

# Hyper-parameters
embedding_dim = 100
input_size = vocab_size
output_size = vocab_size
seq_length = 25
step_size = 3   # shift of sequences

hidden_size = 100
batch_size = 128
num_layers = 8
num_epochs = 1
learning_rate = 0.001

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
# batch_size = N, seq_length = L, D=1 (directions), Hin = size of the input of the model, i.e. embedding_dim,
# not "input_size" defined above. Hout = hidden_size


def one_hot(tensor):    # one-hot encoding (1 of k representation)
    return (F.one_hot(tensor, num_classes=vocab_size)).to(torch.float32)


"""def embedding(tensor):
    embed = nn.Embedding(vocab_size, embedding_dim)  # 37 chars in vocab, 100 dimensional embeddings
    return embed(tensor)"""


def index_from_hot(encoded_tensor):     # returns index of encoded character
    return torch.argmax(encoded_tensor, dim=0)


def char_from_hot(encoded_tensor):      # returns character
    return ix_to_char[torch.argmax(encoded_tensor, dim=0).item()]


"""def test_initialize_seq(train=True):  # initialize inputs and targets sequences
    k = int(0.8 * fulltext_len)
    if train:
        text = encoded_text[:k]     # train, 80%
    else:
        text = encoded_text[k:]     # evaluate, 20%

    # print(f'{encoded_text[k-10:k]} \n{encoded_text[k-1:k+2]}')
    inputs = []     # input sequences
    targets = []    # target characters for each input seq

    text_len = len(text)
    for i in range(0, text_len - seq_length, step_size):
        inputs.append(text[i: i+seq_length])
        targets.append(text[i + seq_length])

    input_tensor = torch.tensor(inputs)     # torch.Size([dim, L]). dim = # of sequences obtained
    target_tensor = torch.tensor(targets)   # torch.Size([dim]).
    input_hot = one_hot(input_tensor)   # torch.Size([dim, L, Hin]). dim = # of sequences obtained
    target_hot = one_hot(target_tensor)     # torch.Size([dim, Hout]).
    # print(f'input tensor dim: {input_hot.size()}\ntarget tensor dim: {target_hot.size()}')
    num_seq = input_hot.size(0)  # dim
    # split input and target into batches of dimension batch_size
    num_batches = int(float(num_seq-batch_size) / float(batch_size))
    # we have num_batches input batches, each of these has length batch_size, and each element is a sequence
    # of length seq_length. each char of the sequence has dimension 37 (input_size)
    batched_input = torch.zeros(num_batches, batch_size, seq_length, input_size)
    # for the target we lack seq_length dimension since it's just a series of one char
    batched_target = torch.zeros(num_batches, batch_size, input_size)

    b = 0   # batch index
    seq_range = 0   # range of sequences to take from input in order to fill batch b
    while b < 4:  # < num_batches:
        s = 0   # index for batch element
        for seq in range(seq_range, seq_range + batch_size, 1):
            print(f'batch={b}')
            # Each batch element will be a sequence from input
            print(f'sequence #{seq}, i.e. input hot[{seq}][:][:] =')
            for i in range(seq_length):
                print(f'{char_from_hot(input_hot[seq][i])}')

            print(f'has to be put as batch #{b} {s}-th element')
            batched_input[b] = input_hot[seq]     # fill b-th batch with batch_size sequences
            batched_target[b] = target_hot[seq]   # and select target
            for i in range(seq_length):
                print(f'batched input: {char_from_hot(batched_input[b][s][i])}')
            print(f'target ---> {char_from_hot(batched_target[b][s])}')
            s += 1
        b += 1  # batch filled: go to new batch
        if seq_range < num_seq - batch_size:  # < num_seq - batch_size
            seq_range += batch_size  # but fill it with 100 new sequences (batch_size = 100)
        else:
            break
    return batched_input, batched_target"""


def initialize_seq(train=True):  # initialize inputs and targets sequences
    if train:
        print(f'initializing training sequences...')
    else:
        print(f'initializing validation sequences...')
    K = int(0.8 * fulltext_len)
    if train:
        text = encoded_text[:K]     # train, 80%
    else:
        text = encoded_text[K:]     # evaluate, 20%

    inputs = []     # input sequences
    targets = []    # target characters for each input seq

    text_len = len(text)
    for i in range(0, text_len - seq_length, step_size):
        inputs.append(text[i: i+seq_length])
        targets.append(text[i + seq_length])

    input_tensor = torch.tensor(inputs, dtype=torch.int64)     # torch.Size([dim, L]). dim = # of sequences obtained
    target_tensor = torch.tensor(targets, dtype=torch.int64)   # torch.Size([dim]).
    num_seq = input_tensor.size(0)  # dim

    # split input and target into batches of dimension batch_size
    num_batches = int(float(num_seq-batch_size) / float(batch_size))
    # we have num_batches input batches, each of these has length batch_size, and each element is a sequence
    # of length seq_length.
    batched_input = torch.zeros(num_batches, batch_size, seq_length, dtype=torch.int64)
    # for the target we lack seq_length dimension since it's just a series of one char
    batched_target = torch.zeros(num_batches, batch_size, dtype=torch.int64)

    b = 0   # batch index
    seq_range = 0   # range of sequences to take from input in order to fill batch b
    while b < num_batches:
        s = 0   # index for batch element
        for seq in range(seq_range, seq_range + batch_size, 1):
            # Each batch element will be a sequence from input
            batched_input[b] = input_tensor[seq]
            batched_target[b] = target_tensor[seq]
            s += 1
        b += 1  # batch filled: go to new batch
        if seq_range < num_seq - batch_size:
            seq_range += batch_size  # but fill it with 100 new sequences (batch_size = 100)
        else:
            break

    return batched_input, batched_target


def test_index_from_hot():
    char = 'n'
    ix = torch.tensor(char_to_ix[char])
    enc = one_hot(ix)
    assert(char == ix_to_char[index_from_hot(enc).item()])


# initialize inputs and targets for train and validation
Xtr, Ytr = initialize_seq()     # training set
Xev, Yev = initialize_seq(train=False)      # validation set
# print(f'Xtr: {Xtr.size()}, Ytr: {Ytr.size()},\nXev: {Xev.size()}, Yev:{Yev.size()}')

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer. Change it to nonlinear

    def forward(self, x):
        # x.size(0) == batch_size
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # not needed, it's by default

        # input shape: (N, L, Hin) or (L, Hin) for unbatched input
        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        out, hidden_state = self.rnn(x_emb)
        # out is (batch_size, seq_length, hidden_size) or (seq_length, hidden_size) for unbatched

        # out: (N, L, Hout). But we don't need all the chars of the sequence, just the last one
        if x.size(0) > 1:
            out = out[:, -1, :]     # select last char: from (N, L, Hout) --> output dim: (N, Hout)

        out = self.fc(out)      # logits
        return out, hidden_state

    def sample(self, seed):     # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            """if len(seed) <= 1:  # if seed is a single char
                seed = one_hot(torch.tensor(char_to_ix[seed]))
                seed = torch.unsqueeze(seed, 0)  # add dummy dimension for matching size
            else:
                seed = one_hot(torch.tensor(encode(seed)))
            # last_state = last_state[:, -1, :]  # squeeze hidden"""
            seed = self.embedding(torch.tensor(encode(seed)))
            seed = self.layer_norm(seed)
            output, _ = self.rnn(seed)
            output = output[-1, :]   # select last char probabilities
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            # sample_ix = prob.argmax().item()
            # print(f'output[-1, :] = {output}')
            # print(f'prob = {prob}')
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            # print(f'maximum probability character following char {char_from_hot(seed[-1])} is {ix_to_char[sample_ix]}')
            # print(f'prob size: {prob.size()}')
            # print(f'sample_ix: {sample_ix}')
            return ix_to_char[sample_ix]


"""class LSTM(nn.Module):
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
            out = out[:, -1, :]     # select last char: from (N, L, Hout) --> output dim: (N, Hout)

        out = self.fc(out)      # logits
        return out, hidden_state

    def sample(self, seed):     # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            if len(seed) <= 1:  # if seed is a single char
                seed = one_hot(torch.tensor(char_to_ix[seed]))
                seed = torch.unsqueeze(seed, 0)  # add dummy dimension for matching size
            else:
                seed = one_hot(torch.tensor(encode(seed)))
            # last_state = last_state[:, -1, :]  # squeeze hidden
            output, _ = self.lstm(seed)
            output = output[-1, :]   # select last char probabilities
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            # sample_ix = prob.argmax().item()
            # print(f'output[-1, :] = {output}')
            # print(f'prob = {prob}')
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            # print(f'maximum probability character following char {char_from_hot(seed[-1])} is {ix_to_char[sample_ix]}')
            # print(f'prob size: {prob.size()}')
            # print(f'sample_ix: {sample_ix}')
            return ix_to_char[sample_ix]"""

# model
model = RNN(embedding_dim, hidden_size, num_layers).to(device)


# test embedding
def test_embedding():
    test_seq = Xtr[0][0]
    test_emb = model.embedding(test_seq)
    print(f'embedding of test sequence: {test_emb}\nwith size {test_emb.size()}')   # ok
    # try to embed single char
    test_char = 'n'
    char_emb = model.embedding(torch.tensor(char_to_ix[test_char], dtype=torch.long))
    print(f'embedding of single char: {char_emb}\nwith size {char_emb.size()}')     # ok


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""# is the model working all right?
# i.e. is the output actually logits? (log of probabilities of the characters)
# take a letter as a seed:
test_char = 's'
char_hot = one_hot(torch.tensor(char_to_ix[test_char]))
x_test = torch.unsqueeze(char_hot, 0)
# print(f'encoded character tensor is {x_test} and is shaped as {x_test.size()}')
logit, h = model.forward(x_test)
# print(f'logits: {logit}\nlogits shape: {logit.size()}')
prob = F.softmax(logit, dim=1)
# print(f'probabilities:{prob} are summed to {torch.sum(prob)}')
char_ixs = range(vocab_size)
probabilities = [0]*vocab_size
for i in range(vocab_size):
    probabilities[i] = prob[0][i].item()
# print(f'{probabilities}, char_ixs: {char_ixs}')"""


# training
n_iter = 10000
current_loss_tr = 0
current_loss_ev = 0
tr_losses = []
ev_losses = []
plot_steps = n_iter / 50
sample_steps = n_iter / 50

tr_batches = Xtr.size(0)
ev_batches = Xev.size(0)

print('starting training and evaluation...')
for epoch in range(num_epochs):
    for i in range(n_iter):
        # training
        # select random batch:
        b_tr = random.randint(0, tr_batches - 1)
        b_ev = random.randint(0, ev_batches - 1)

        # Xb = layer_norm(Xtr[b_tr])
        Xb = Xtr[b_tr]
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
            # Xevb = layer_norm(Xev[b_ev])
            Xevb = Xev[b_ev]
            Yevb = Yev[b_ev]
            out, _ = model(Xevb)
            ev_loss = criterion(out, Yevb)
            current_loss_ev += ev_loss.item()

        if (i + 1) % plot_steps == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_iter}], train Loss: {tr_loss.item():.4f}, eval Loss: {ev_loss.item():.4f}')

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




"""new_logit, hidden = model.forward(x_test)
new_prob = F.softmax(new_logit, dim=1)
char_ixs = range(vocab_size)
new_probabilities = [0]*vocab_size
for i in range(vocab_size):
    new_probabilities[i] = new_prob[0][i].item()

fig, axis = plt.subplots(1, 2)
fig.tight_layout()
axis[0].set_xlabel('character indexes')
axis[1].set_xlabel('character indexes')
axis[0].set_ylabel(f'probabilities for each character to follow {test_char} before training')
axis[1].set_ylabel(f'probabilities for each character to follow {test_char} after training')
axis[0].bar(char_ixs, probabilities, color='blue')
axis[1].bar(char_ixs, new_probabilities, color='green')
plt.show()

# now, is sampling all right?
# sample through a multinomial with softmax
test_hist = [0]*vocab_size
for i in range(100):
    sample_ix = torch.multinomial(new_prob, 1, replacement=True).item()
    # print(f'sampled index from true distribution of {test_char} is: {ix_to_char[sample_ix]}')
    test_hist[sample_ix] += 1"""

# how about sampling with a given hidden state and not with all zeros?
seed_seq = 'nel mezzo del cammin di nostra vita'
sample_seq = [c for c in seed_seq]
sample_len = 75
for k in range(sample_len):
    # print(f'seed sequence: {sample_seq}')
    prediction = model.sample(sample_seq)
    # print(f'predicted char: {prediction}')
    sample_seq.append(prediction)
txt = ''.join(sample_seq)
print(f'sampled text: {txt}')

"""# print(f'{h_n}\n{h_n.size()}')
memory_out, state = model.rnn(x_test, h_n)    # hn last hidden state from training
memory_logits = model.fc(memory_out)
memory_prob = F.softmax(memory_logits, dim=1)

memory_hist = [0]*vocab_size
for i in range(100):
    sample_ix = torch.multinomial(new_prob, 1, replacement=True).item()
    # print(f'sampled index from true distribution of {test_char} is: {ix_to_char[sample_ix]}')
    memory_hist[sample_ix] += 1

subtraction = [0]*vocab_size    # difference in occurences between memory and no memory
for i in range(vocab_size):
    subtraction[i] = abs(memory_hist[i] - test_hist[i])

fig1, axis1 = plt.subplots(1, 4, sharey=True)
fig1.tight_layout()
axis1[0].set_xlabel('character indexes')
axis1[1].set_xlabel('character indexes')
axis1[2].set_xlabel('character indexes')
axis1[3].set_xlabel('character indexes')
axis1[0].set_ylabel(f'occurences of sampled characters following {test_char} without memory')
axis1[1].set_ylabel(f'occurences of sampled characters following {test_char} with memory')
axis1[2].set_ylabel(f'difference between sampling with memory and without memory')
axis1[3].set_ylabel(f'probabilities for each character to follow {test_char} after training')
axis1[0].bar(char_ixs, test_hist, color='red')
axis1[1].bar(char_ixs, memory_hist, color='green')
axis1[2].bar(char_ixs, subtraction, color='blue')
axis1[3].bar(char_ixs, new_probabilities, color='orange', sharey=False)
plt.show()"""

# very small difference, doesn't change much between memory and no memory

end = time.time()
elapsed_time = end - start

print(f'elapsed time in process: {int(elapsed_time/60)} minutes.\n****list of hyperparameters used:****\nembedding_dim = {embedding_dim},\nseq_length = {seq_length},\nstep_size =  {step_size},\nhidden_size = {hidden_size}, \nbatch_size = {batch_size},\nnum_layers = {num_layers},\nnum_epochs = {num_epochs},\nlearning rate = {learning_rate}')



