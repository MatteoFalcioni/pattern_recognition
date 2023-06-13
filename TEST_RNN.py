import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import time
from scipy.spatial import distance
import configparser

import models

start = time.time()

print('please, insert the configuration file name in order to get the hyperparameters. If you want to use default hyperparameters, type: default')
config = configparser.ConfigParser()
config_filename = input()
if config_filename.lower() == 'default':
    config.read('configuration.txt')
else:
    config.read(config_filename)

# fix the seed

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
model_choice = input().upper()
while model_choice != 'GRU' and model_choice != 'LSTM' and model_choice != 'RNN':
    print(f'model {model_choice} is an invalid keyword. Choose between RNN, LSTM or GRU')
    model_choice = input()

print('All right! Now, type train if you want to train the model, type generate if you just want to generate text')
TRAIN = input().lower()
while TRAIN != 'train' and TRAIN != 'generate':
    print(f'model {TRAIN} is an invalid keyword. Choose between train or generate.')
    TRAIN = input()

# hyperparameters
SEQ_LENGTH = int(config.get('Hyperparameters', 'SEQ_LENGTH'))
STEP_SIZE = int(config.get('Hyperparameters', 'STEP_SIZE'))
EMBEDDING_DIM = int(config.get('Hyperparameters', 'EMBEDDING_DIM'))
HIDDEN_SIZE = int(config.get('Hyperparameters', 'HIDDEN_SIZE'))
BATCH_SIZE = int(config.get('Hyperparameters', 'BATCH_SIZE'))
NUM_LAYERS = int(config.get('Hyperparameters', 'NUM_LAYERS'))
DECAY_RATE = float(config.get('Hyperparameters', 'DECAY_RATE'))
NUM_EPOCHS = int(config.get('Hyperparameters', 'NUM_EPOCHS'))

if model_choice == 'RNN':
    LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE_RNN'))   # 0.1 works alright for RNN until epoch 12, 0.5 seems to work better for LSTM until epoch 25. 0.5 works for gru until 10
    DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP_RNN'))
if model_choice == 'LSTM':
    LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE_LSTM'))
    DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP_LSTM'))
if model_choice == 'GRU':
    LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE_GRU'))
    DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP_GRU'))

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
    test_enc = encode(fulltext[:200])
    test_input, test_target = initialize_seq(test, SEQ_LENGTH, STEP_SIZE)

    for i in range(0, len(test) - SEQ_LENGTH, STEP_SIZE):
        for k in range(SEQ_LENGTH):
            assert(test_enc[i: i + k] == test_input[i][i: i + k])
        assert(test_enc[i + SEQ_LENGTH] == test_target[i])


tr_inputs, tr_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE)
ev_inputs, ev_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE, train=False)
tr_dataset = models.DemandDataset(torch.tensor(tr_inputs).to(device), torch.tensor(tr_targets).to(device))
ev_dataset = models.DemandDataset(torch.tensor(ev_inputs).to(device), torch.tensor(ev_targets).to(device))
tr_dataloader = models.DataLoader(tr_dataset, shuffle=True, batch_size=BATCH_SIZE)
ev_dataloader = models.DataLoader(ev_dataset, shuffle=True, batch_size=BATCH_SIZE)
n_train = len(tr_dataloader)
n_eval = len(ev_dataloader)


def test_sample(system):

    test_seq = 'nel mezzo del cammin di nostra vita'
    test_seed = system.embedding(torch.tensor(encode(test_seq)))
    output, _ = system.rnn(test_seed)
    output = output[-1, :]  # select last char probabilities
    logits = system.fc(output)
    prob = F.softmax(logits, dim=0)
    probabilities = [0.0] * vocab_size
    for i in range(vocab_size):
        probabilities[i] = prob[i].item()

    test_samples = [0.0] * vocab_size
    subtraction = [0.0] * vocab_size
    n_iter = 200000
    for i in range(n_iter):
        test_prediction = model.sample(test_seq)
        test_samples[char_to_ix[test_prediction]] += 1.0/n_iter
    # sampled chars distribution should tend to prob distribution for N --> infinity
    for k in range(vocab_size):
        subtraction[k] = abs(test_samples[k] - probabilities[k])
        assert(subtraction[k] < 0.001)


"""def test_accuracy():
    model.accuracy(ev_inputs, ev_targets) * 100"""


if model_choice == 'RNN':
    model = models.RNN(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
if model_choice == 'LSTM':
    model = models.LSTM(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
if model_choice == 'GRU':
    model = models.GRU(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)


if TRAIN == 'generate':
    state_dict = torch.load(f'pretrained/{model_choice}.pth')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()    # models are saved in train mode


def distances(sample, true_seq):
    seq_len = len(sample)
    sample = encode(sample)     # distance expects arrays, not strings
    true_seq = encode(true_seq)
    d1 = round(distance.hamming(sample, true_seq) * seq_len)
    d2 = distance.cosine(sample, true_seq)
    return d1, d2


def test_distances(system):
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
    assert(d1 > 0)
    assert(d2 > 0)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# implementing lr decay through epochs :
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)

previous_val_loss = float('inf')
epoch_tr_loss = 0
epoch_ev_loss = 0
epoch_tr_losses = []
epoch_ev_losses = []
epoch_perplexities = []

min_epochs = 30

if TRAIN == 'train':
    print('starting training and evaluation...')
    for epoch in range(NUM_EPOCHS):
        print(f'epoch [{epoch}/{NUM_EPOCHS}]')
        for X, y in tr_dataloader:
            # training
            model.train()

            # forward pass
            outputs, h_n = model(X)
            tr_loss = criterion(outputs, y)

            # backward and optimize
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            epoch_tr_loss += tr_loss.item() / float(n_train)

        epoch_tr_losses.append(epoch_tr_loss)
        epoch_tr_loss = 0

        epoch_perplexity = 0
        for X, y in ev_dataloader:
            # evaluation
            model.eval()

            with torch.no_grad():
                # evaluate loss on eval set
                out, _ = model(X)
                ev_loss = criterion(out, y)

                epoch_ev_loss += ev_loss.item() / float(n_eval)
                epoch_perplexity += math.exp(ev_loss.item())/float(n_eval)  # 'average' perplexity

        epoch_ev_losses.append(epoch_ev_loss)
        epoch_ev_loss = 0
        epoch_perplexities.append(epoch_perplexity)
        epoch_perplexity = 0

        # Check for overfitting
        if epoch_ev_loss >= previous_val_loss and epoch > min_epochs:
            print("Overfitting detected! Stopping the training loop...")
            break

        # Update the previous validation loss variable
        previous_val_loss = epoch_ev_loss

        scheduler.step()    # lr = lr*0.1 after decay_step steps

        print(f'avg epoch #{epoch} train loss: {epoch_tr_losses[epoch]}\navg epoch #{epoch} validation loss: {epoch_ev_losses[epoch]}')

        tr_losses = []
        ev_losses = []

    # write losses on separate file
    file_toplot = f'{model_choice}_toplot'
    with open(file_toplot, 'w') as file:
        # Zip the lists and iterate over the pairs
        for tr_loss, ev_loss, perplexity in zip(epoch_tr_losses, epoch_ev_losses, epoch_perplexities):
            # Write the values to the file with a space in between
            file.write(f'{tr_loss}\t{ev_loss}\t{perplexity}\n')

seed_seq = 'nel mezzo del cammin di nostra vita'
sample_seq = [c for c in seed_seq]
sample_len = 250
for step in range(sample_len):
    prediction = model.sample(sample_seq)
    sample_seq.append(prediction)
sampled_txt = ''.join(sample_seq)
print(f'sampled text: {sampled_txt}')
true_text = fulltext[:len(sample_seq)]

hamming_d, cosine_d = distances(sampled_txt, true_text)
accuracy = model.accuracy(ev_inputs, ev_targets)*100

end = time.time()
elapsed_time = end - start
with open('efficiency.txt', 'a') as file:
    file.write(f'{model_choice}\t{hamming_d}\t{cosine_d}\t{accuracy}\t{elapsed_time}')

print(f'computing distances between sampled string and real string:\nhamming distance = {hamming_d}\ncosine '
      f'similarity = {cosine_d}')

print(f'elapsed time in process: {int(elapsed_time/60)} minutes.')

if TRAIN == 'train':
    print('do you want to save the trained model? Type yes or no')
    CHOICE = input()
    while CHOICE != 'yes' and CHOICE != 'no':
        print('this key is not existing, type yes or no to choose whether to save the model or not')
        CHOICE = input()

    if CHOICE == 'yes':
        # saving the model's state_dict
        PATH = f'pretrained/{model_choice}.pth'
        torch.save(model.state_dict(), PATH)    # this is saved in train mode. to use it, put it back to eval with .eval()
        print('model has been saved successfully')

    # when you re-create the model, do the following:
    # (i) set up model: for example, model=RNN(...)
    # (ii) load the saved parameters: model.load_state_dict(torch.load(FILE))
    # (iii) send it to GPU: model.to(device)"""





