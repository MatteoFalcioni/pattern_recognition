import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
import sys
from sys import argv
import configparser

start = time.time()

print('please, insert the configuration file name in order to get the hyperparameters. If you want to use default hyperparameters, type: default')
config = configparser.ConfigParser()
config_filename = input()
#config_filename = config.read(sys.argv[1])
if config_filename == 'default':
    config.read('configuration.txt')

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

print('please, insert the RNN model you want to use. Choices are between: RNN, LSTM, GRU')
model_choice = input()
# model_choice = config.read(sys.argv[1])
if model_choice != 'GRU' and model_choice != 'LSTM' and model_choice != 'RNN':
    raise ValueError(f'model {model_choice} does not exist. Please insert one between RNN, LSTM, GRU')

print('All right! Now, type train if you want to train the model, type generate if you just want to generate text')
TRAIN = input()
if TRAIN != 'train' and TRAIN != 'generate':
    raise ValueError(f'model {model_choice} is an invalid keyword. Choose between train or generate.')

# hyperparameters
SEQ_LENGTH = int(config.get('Hyperparameters', 'SEQ_LENGTH'))
STEP_SIZE = int(config.get('Hyperparameters', 'STEP_SIZE'))     # shift of sequences
EMBEDDING_DIM = int(config.get('Hyperparameters', 'EMBEDDING_DIM'))
HIDDEN_SIZE = int(config.get('Hyperparameters', 'HIDDEN_SIZE'))
BATCH_SIZE = int(config.get('Hyperparameters', 'BATCH_SIZE'))
NUM_LAYERS = int(config.get('Hyperparameters', 'NUM_LAYERS'))
NUM_EPOCHS = int(config.get('Hyperparameters', 'NUM_EPOCHS'))
LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE'))   # 0.1 works alright for RNN until epoch 12, 0.5 seems to work better for LSTM until epoch 25. 0.5 works for gru until 10
DECAY_RATE = float(config.get('Hyperparameters', 'DECAY_RATE'))
DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP'))

INPUT_SIZE = vocab_size
OUTPUT_SIZE = vocab_size


# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
# batch_size = N, seq_length = L, D=1 (directions), Hin = size of the input of the model, i.e. embedding_dim


def initialize_seq(corpus, seq_length, step_size, train=True):
    encoded_text = encode(corpus)
    k = int(0.8 * len(corpus))
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


def test_initialize_seq():
    test = fulltext[:200]
    test_input, test_target = initialize_seq(test, SEQ_LENGTH, STEP_SIZE)
    for i in range(0, len(test) - SEQ_LENGTH, STEP_SIZE):
        assert(test[i: i + SEQ_LENGTH] == test_input[i: i + SEQ_LENGTH])
        assert(test[i + SEQ_LENGTH] == test_target[i])


test_initialize_seq()


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


tr_inputs, tr_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE)
ev_inputs, ev_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE, train=False)
tr_dataset = DemandDataset(torch.tensor(tr_inputs).to(device), torch.tensor(tr_targets).to(device))
ev_dataset = DemandDataset(torch.tensor(ev_inputs).to(device), torch.tensor(ev_targets).to(device))
tr_dataloader = DataLoader(tr_dataset, shuffle=True, batch_size=BATCH_SIZE)
ev_dataloader = DataLoader(ev_dataset, shuffle=True, batch_size=BATCH_SIZE)
n_train = len(tr_dataloader)
n_eval = len(ev_dataloader)


class RNN(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_size, num_layers):
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
        # x_emb = self.layer_norm(x_emb)
        out, hidden_state = self.rnn(x_emb)

        if x.size(0) > 1:
            out = out[:, -1, :]

        out = self.fc(out)
        return out, hidden_state

    def sample(self, seed):     # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            seed = self.embedding(torch.tensor(encode(seed)))
            output, _ = self.rnn(seed)
            output = output[-1, :]   # select last char probabilities
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            return ix_to_char[sample_ix]

    def accuracy(self, input_seqs, targets):    # probably not needed
        accuracy = 0
        num_seqs = len(input_seqs)
        for i in range(num_seqs):
            predicted_char = self.sample(decode(input_seqs[i]))
            actual_char = ix_to_char[targets[i]]
            if actual_char == predicted_char:
                accuracy += 1.0/float(num_seqs)
        return accuracy


def test_sample():
    return 'test sample'


def test_accuracy():
    return 'test accuracy'


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
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


class GRU(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim,  hidden_size, num_layers):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        out, hidden_state = self.gru(x_emb)
        if x.size(0) > 1:
            out = out[:, -1, :]

        out = self.fc(out)
        return out, hidden_state

    def sample(self, seed):     # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            seed = self.embedding(torch.tensor(encode(seed)))
            output, _ = self.gru(seed)
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
    model = RNN(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
if model_choice == 'LSTM':
    model = LSTM(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
if model_choice == 'GRU':
    model = GRU(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)

if TRAIN == 'generate':
    file_to_load = f'{model_choice}.pth'
    model.load_state_dict(torch.load(file_to_load))


def similarity(sample, true_seq, distance_type):
    seq_len = len(sample)
    sample = encode(sample)     # distance expects arrays, not strings
    true_seq = encode(true_seq)
    if distance_type == 'hamming':
        d = round(distance.hamming(sample, true_seq) * seq_len)
    if distance_type == 'cosine':
        d = distance.cosine(sample, true_seq)
    return d


def test_similarity(system):
    seed_seq = 'nel mezzo del cammin di nostra vita'
    sample_seq = [c for c in seed_seq]
    sample_len = 50
    for step in range(sample_len):
        prediction = system.sample(sample_seq)
        sample_seq.append(prediction)

    seq1 = sample_seq
    seq2 = fulltext[:len(sample_seq)]
    seq1_enc = encode(seq1)
    seq2_enc = encode(seq2)
    assert(len(seq1_enc) == len(seq2_enc))

    d0_ham = distance.hamming(seq1_enc, seq1_enc)
    d0_cos = distance.cosine(seq1_enc, seq1_enc)

    assert(d0_ham < 0.0001)
    assert(d0_cos < 0.0001)

    d1 = round(distance.hamming(seq1_enc, seq2_enc) * len(seq1_enc))
    d2 = distance.cosine(seq1_enc, seq2_enc)

    return d1, d2


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# implementing lr decay through epochs :
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)

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
for epoch in range(NUM_EPOCHS):
    print(f'epoch [{epoch}/{NUM_EPOCHS}]')
    for X, y in tr_dataloader:
        # training
        model.train()

        # tr_step += 1
        # forward pass
        outputs, h_n = model(X)
        tr_loss = criterion(outputs, y)

        # backward and optimize
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # current_loss_tr += tr_loss.item()
        epoch_tr_loss += tr_loss.item() / float(n_train)

        # if (tr_step + 1) % plot_steps == 0:
        #    tr_losses.append(current_loss_tr / plot_steps)
        # current_loss_tr = 0
    # tr_step = 0
    epoch_tr_losses.append(epoch_tr_loss)
    epoch_tr_loss = 0

    for X, y in ev_dataloader:
        # evaluation
        model.eval()

        # ev_step += 1
        with torch.no_grad():
            # evaluate loss on eval set
            out, _ = model(X)
            ev_loss = criterion(out, y)

            # current_loss_ev += ev_loss.item()
            epoch_ev_loss += ev_loss.item() / float(n_eval)

            # if (ev_step + 1) % 3 == 0:
            #     ev_losses.append(current_loss_ev / 3)
            # current_loss_ev = 0
    # ev_step = 0
    epoch_ev_losses.append(epoch_ev_loss)
    epoch_ev_loss = 0

    scheduler.step()    # lr = lr*0.1

    print(f'avg epoch #{epoch} train loss: {epoch_tr_losses[epoch]}\navg epoch #{epoch} validation loss: {epoch_ev_losses[epoch]}')

    tr_losses = []
    ev_losses = []

plt.figure()
plt.plot(epoch_tr_losses, color='blue', label='training loss')
plt.plot(epoch_ev_losses, color='orange', label='evaluation loss')
plt.title(f'Training loss vs evaluation loss over {NUM_EPOCHS} epochs')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

seed_seq = 'nel mezzo del cammin di nostra vita'
sample_seq = [c for c in seed_seq]
sample_len = 250
for step in range(sample_len):
    prediction = model.sample(sample_seq)
    sample_seq.append(prediction)
sampled_txt = ''.join(sample_seq)
print(f'sampled text: {sampled_txt}')

true_text = fulltext[:len(sample_seq)]

hamming_d = similarity(sampled_txt, true_text, 'hamming')
cosine_d = similarity(sampled_txt, true_text, 'cosine')

print(f'computing distances between sampled string and real string:\nhamming distance = {hamming_d}\ncosine '
      f'similarity = {cosine_d}')

# print(f'accuracy = {model.accuracy(ev_inputs, ev_targets)*100}%')

end = time.time()
elapsed_time = end - start

print(f'elapsed time in process: {int(elapsed_time/60)} minutes.\n***** list of hyperparameters '
      f'used: *****\nembedding_dim = {EMBEDDING_DIM},\nseq_length = {SEQ_LENGTH},\nstep_size =  {STEP_SIZE},'
      f'\nhidden_size = {HIDDEN_SIZE}, \nbatch_size = {BATCH_SIZE},\nnum_layers = {NUM_LAYERS},\nnum_epochs = '
      f'{NUM_EPOCHS},\nlearning rate = {LEARNING_RATE},\nlr decay factor={DECAY_RATE}\nlr decay step={DECAY_STEP}')

# saving the model
FILE = f'{model_choice}.pth'
torch.save(model, FILE)    # this is saved in train mode. to use it, put it back to eval with .eval()
# when you re-create the model, do the following:
# (i) set up model: for example, model=RNN(...)
# (ii) load the saved parameters: model.load_state_dict(torch.load(FILE))
# (iii) send it to GPU: model.to(device)"""





