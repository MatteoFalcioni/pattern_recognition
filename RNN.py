import torch
import torch.nn as nn
import math
import time
import configparser
from torch.utils.data import DataLoader

import models
import data_config
from data_config import initialize_seq, vocab_size, fulltext

# Device config. Use GPU if possible, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix the seed for reproducibility
torch.manual_seed(1234567890)

# ----------------------------------- user interface ----------------------------------- #
print('please, insert the configuration file name in order to get the hyperparameters. If you want to use default '
      'hyperparameters, type: default')
config = configparser.ConfigParser()
config_filename = input()
if config_filename.lower() == 'default':
    config.read('configuration.txt')
else:
    config.read(config_filename)

print('please, insert the RNN model you want to use. Choices are between: RNN, LSTM, GRU')
model_choice = input().upper()
while model_choice != 'GRU' and model_choice != 'LSTM' and model_choice != 'RNN':
    print(f'model {model_choice} is an invalid keyword. Choose between RNN, LSTM or GRU')
    model_choice = input()

print('All right! Now, type train if you want to train the model, type generate if you just want to generate text')
TRAIN = input().lower()
while TRAIN != 'train' and TRAIN != 'generate':
    print(f'model {TRAIN} is an invalid keyword. Choose between train or generate.')
    TRAIN = input().lower()
# ----------------------------------------------------------------------------------------

# get hyperparameters from configuration file
SEQ_LENGTH = int(config.get('Hyperparameters', 'SEQ_LENGTH'))
STEP_SIZE = int(config.get('Hyperparameters', 'STEP_SIZE'))
EMBEDDING_DIM = int(config.get('Hyperparameters', 'EMBEDDING_DIM'))
HIDDEN_SIZE = int(config.get('Hyperparameters', 'HIDDEN_SIZE'))
BATCH_SIZE = int(config.get('Hyperparameters', 'BATCH_SIZE'))
NUM_LAYERS = int(config.get('Hyperparameters', 'NUM_LAYERS'))
DECAY_RATE = float(config.get('Hyperparameters', 'DECAY_RATE'))
NUM_EPOCHS = int(config.get('Hyperparameters', 'NUM_EPOCHS'))

INPUT_SIZE = vocab_size
OUTPUT_SIZE = vocab_size

if model_choice == 'RNN':
    LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE_RNN'))
    DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP_RNN'))
    MIN_EPOCHS = int(config.get('Hyperparameters', 'MIN_EPOCHS_RNN'))
    model = models.RNN(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
elif model_choice == 'LSTM':
    LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE_LSTM'))
    DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP_LSTM'))
    MIN_EPOCHS = int(config.get('Hyperparameters', 'MIN_EPOCHS_LSTM'))
    model = models.LSTM(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
elif model_choice == 'GRU':
    LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE_GRU'))
    DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP_GRU'))
    MIN_EPOCHS = int(config.get('Hyperparameters', 'MIN_EPOCHS_GRU'))
    model = models.GRU(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)


# initialize inputs and targets for train and validation
tr_inputs, tr_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE)
ev_inputs, ev_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE, train=False)
tr_dataset = data_config.DemandDataset(torch.tensor(tr_inputs).to(device), torch.tensor(tr_targets).to(device))
ev_dataset = data_config.DemandDataset(torch.tensor(ev_inputs).to(device), torch.tensor(ev_targets).to(device))
tr_dataloader = DataLoader(tr_dataset, shuffle=True, batch_size=BATCH_SIZE)     # training set (80% of data)
ev_dataloader = DataLoader(ev_dataset, shuffle=True, batch_size=BATCH_SIZE)     # validation set (20% of data)
n_train = len(tr_dataloader)
n_eval = len(ev_dataloader)


if TRAIN == 'generate':
    state_dict = torch.load(f'pretrained/{model_choice}.pth')
    model.load_state_dict(state_dict)   # load pre-trained state dict
    model.to(device)    # send it to gpu
    model.eval()    # models are saved in train mode


if TRAIN == 'train':
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)   # SGD works better than Adam for this task
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)  # lr decay through epochs

    previous_val_loss = float('inf')
    epoch_tr_loss = 0
    epoch_ev_loss = 0
    epoch_tr_losses = []
    epoch_ev_losses = []
    epoch_perplexities = []

    start = time.time()
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
        if epoch_ev_loss >= previous_val_loss and epoch > MIN_EPOCHS:
            print("Overfitting detected! Stopping the training loop...")
            break

        # Update the previous validation loss variable
        previous_val_loss = epoch_ev_loss

        scheduler.step()    # lr = lr*DECAY_RATE after DECAY_STEP steps

        print(f'avg epoch #{epoch} train loss: {epoch_tr_losses[epoch]}\navg epoch #{epoch} validation loss: {epoch_ev_losses[epoch]}')
        tr_losses = []
        ev_losses = []

    # write losses to file
    file_toplot = f'toplot/{model_choice}_toplot.txt'
    with open(file_toplot, 'w') as file:
        # Zip the lists and iterate over the pairs
        for tr_loss, ev_loss, perplexity in zip(epoch_tr_losses, epoch_ev_losses, epoch_perplexities):
            # Write the values to the file with a space in between
            file.write(f'{tr_loss}\t{ev_loss}\t{perplexity}\n')

    end = time.time()
    training_time = end - start

    with open('toplot/efficiency.txt', 'a') as file:    # write training time to file
        file.write(f'\n{model_choice}\t{training_time}')

    print('do you want to save the trained model? Type yes or no')
    CHOICE = input().lower()
    while CHOICE != 'yes' and CHOICE != 'no':
        print('this key is not existing, type yes or no to choose whether to save the model or not')
        CHOICE = input().lower

    if CHOICE == 'yes':
        # saving the model's state_dict
        PATH = f'pretrained/{model_choice}.pth'
        torch.save(model.state_dict(), PATH)  # this is saved in train mode (!)
        print('model has been saved successfully')

# end of training

# generate (if trained, generates text after training)
seed_seq = 'nel mezzo del cammin di nostra vita'
sample_seq = [c for c in seed_seq]
sample_len = 250
for step in range(sample_len):
    prediction = model.sample(sample_seq)
    sample_seq.append(prediction)
sampled_txt = ''.join(sample_seq)
print(f'sampled text: {sampled_txt}')
true_text = fulltext[:len(sample_seq)]






