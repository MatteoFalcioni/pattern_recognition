import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
fulltext_len, vocab_size = len(fulltext), len(chars)

# build the vocabulary of characters and mappings to / from integers
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_ix[c] for c in s]  # encoder : take a string , output a list of integers
decode = lambda l: ''.join([ix_to_char[i] for i in l])  # decoder : take a list of integers, output a string
encoded_text = encode(fulltext)

model_choice = 'LSTM'

# Hyper-parameters
embedding_dim = 100
input_size = vocab_size
output_size = vocab_size
seq_length = 25
step_size = 3   # shift of sequences

hidden_size = 100
batch_size = 128
num_layers = 5
num_epochs = 25
learning_rate = 0.01
decay_rate = 0.1
decay_step = 3

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
# batch_size = N, seq_length = L, D=1 (directions), Hin = size of the input of the model, i.e. embedding_dim,
# not "input_size" defined above. Hout = hidden_size


def initialize_seq(train=True):
    k = int(0.8 * fulltext_len)
    if train:
        text = encoded_text[:k]     # train, 80%
    else:
        text = encoded_text[k:]     # evaluate, 20%

    inputs = []     # input sequences
    targets = []    # target characters for each input seq

    text_len = len(text)
    for i in range(0, text_len - seq_length, step_size):
        inputs.append(text[i: i+seq_length])
        targets.append(text[i + seq_length])

    return inputs, targets


class DemandDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        data = self.X_train[idx]
        labels = self.y_train[idx]
        return data, labels


tr_inputs, tr_targets = initialize_seq()
ev_inputs, ev_targets = initialize_seq(train=False)
tr_dataset = DemandDataset(torch.tensor(tr_inputs).to(device), torch.tensor(tr_targets).to(device))
ev_dataset = DemandDataset(torch.tensor(ev_inputs).to(device), torch.tensor(ev_targets).to(device))
tr_dataloader = DataLoader(tr_dataset, shuffle=True, batch_size=batch_size)
ev_dataloader = DataLoader(ev_dataset, shuffle=True, batch_size=batch_size)
n_train = len(tr_dataloader)
n_eval = len(ev_dataloader)


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
        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        out, hidden_state = self.rnn(x_emb)

        if x.size(0) > 1:
            out = out[:, -1, :]

        out = self.fc(out)
        return out, hidden_state

    def sample(self, seed):     # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            seed = self.embedding(torch.tensor(encode(seed)))
            # seed = self.layer_norm(seed)
            output, _ = self.rnn(seed)
            output = output[-1, :]   # select last char probabilities
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            return ix_to_char[sample_ix]

    def accuracy(self, input_seqs, targets):
        accuracy = 0
        num_seqs = len(input_seqs)
        for i in range(num_seqs):
            predicted_char = self.sample(decode(input_seqs[i]))
            actual_char = ix_to_char[targets[i]]
            if actual_char == predicted_char:
                accuracy += 1.0/float(num_seqs)
        return accuracy


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        out, hidden_state = self.lstm(x_emb)
        if x.size(0) > 1:
            out = out[:, -1, :]

        out = self.fc(out)
        return out, hidden_state

    def sample(self, seed):     # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            seed = self.embedding(torch.tensor(encode(seed)))
            # seed = self.layer_norm(seed)
            output, _ = self.lstm(seed)
            output = output[-1, :]
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            return ix_to_char[sample_ix]

    def accuracy(self, input_seqs, targets):
        accuracy = 0
        num_seqs = len(input_seqs)
        for i in range(num_seqs):
            predicted_char = self.sample(decode(input_seqs[i]))
            actual_char = ix_to_char[targets[i]]
            if actual_char == predicted_char:
                accuracy += 1.0/float(num_seqs)
        return accuracy


if model_choice == 'RNN':
    model = RNN(embedding_dim, hidden_size, num_layers).to(device)
if model_choice == 'LSTM':
    model = LSTM(embedding_dim, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# implementing lr decay through epochs :
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

current_loss_tr = 0
current_loss_ev = 0
tr_step = 0
ev_step = 0
tr_losses = []
ev_losses = []

epoch_tr_loss = 0
epoch_ev_loss = 0
epoch_tr_losses = []
epoch_ev_losses = []

print('starting training and evaluation...')
for epoch in range(num_epochs):
    print(f'epoch [{epoch}/{num_epochs}]')
    for X, y in tr_dataloader:
        # training
        model.train()

        tr_step += 1
        # forward pass
        outputs, h_n = model(X)
        tr_loss = criterion(outputs, y)

        # backward and optimize
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # current_loss_tr += tr_loss.item()
        epoch_tr_loss += tr_loss.item()*1000 / float(n_train)

        """if (tr_step + 1) % plot_steps == 0:
            tr_losses.append(current_loss_tr / plot_steps)
        current_loss_tr = 0"""
    tr_step = 0
    epoch_tr_losses.append(epoch_tr_loss)
    epoch_tr_loss = 0

    for X, y in ev_dataloader:
        # evaluation
        model.eval()

        ev_step += 1
        with torch.no_grad():
            # evaluate loss on eval set
            out, _ = model(X)
            ev_loss = criterion(out, y)

            # current_loss_ev += ev_loss.item()
            epoch_ev_loss += ev_loss.item()*1000 / float(n_eval)

            """if (ev_step + 1) % 3 == 0:
                ev_losses.append(current_loss_ev / 3)
            current_loss_ev = 0"""
    ev_step = 0
    epoch_ev_losses.append(epoch_ev_loss)
    epoch_ev_loss = 0

    scheduler.step()    # lr = lr*0.1

    print(f'avg epoch #{epoch} train loss: {epoch_tr_losses[epoch]}\navg epoch #{epoch} validation loss: {epoch_ev_losses[epoch]}')

    tr_losses = []
    ev_losses = []

plt.figure()
plt.plot(epoch_tr_losses, color='blue', label='training loss')
plt.plot(epoch_ev_losses, color='orange', label='evaluation loss')
plt.title(f'Training loss vs evaluation loss over {num_epochs} epochs')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

seed_seq = 'nel mezzo del cammin di nostra vita'
sample_seq = [c for c in seed_seq]
sample_len = 120
for k in range(sample_len):
    prediction = model.sample(sample_seq)
    sample_seq.append(prediction)
txt = ''.join(sample_seq)
print(f'sampled text: {txt}')

print(f'accuracy = {model.accuracy(ev_inputs, ev_targets)*100}%')

end = time.time()
elapsed_time = end - start

print(f'elapsed time in process: {int(elapsed_time/60)} minutes.\n***** list of hyperparameters '
      f'used: *****\nembedding_dim = {embedding_dim},\nseq_length = {seq_length},\nstep_size =  {step_size},'
      f'\nhidden_size = {hidden_size}, \nbatch_size = {batch_size},\nnum_layers = {num_layers},\nnum_epochs = '
      f'{num_epochs},\nlearning rate = {learning_rate},\nlr decay factor={decay_rate}\nlr decay step={decay_step}')

# saving the model
FILE = f'{model_choice}.pth'
torch.save(model, FILE)    # this is saved in train mode. to use it, put it back to eval with .eval()
# when you re-create the model, do the following:
# (i) set up model: for example, model=RNN(...)
# (ii) load the saved parameters: model.load_state_dict(torch.load(FILE))
# (iii) send it to GPU: model.to(device)






