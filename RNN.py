import torch
import torch.nn as nn
import time
import configparser
from torch.utils.data import DataLoader
import models as mdl
import data_config
from training import train_epochs, save_data, inference
from data_config import initialize_seq, vocab_size, fulltext, get_parser

"""
this file contains the main script to train and generate text from the models.
"""

# Device config. Use GPU if possible, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Command line argument parser. See `data_config.get_parser()`.
parser = get_parser()
args = parser.parse_args()
hyperParam_path = args.CONFIG   # path to the Hyperparameters configuration file
model_choice = args.MODEL   # chosen model
TRAIN = args.TRAINING   # choice between training or inference
SAVE = args.SAVING  # choice whether to save or not the model parameters


# get hyperparameters from configuration file
config = configparser.ConfigParser()
config.read(hyperParam_path)
SEQ_LENGTH = int(config.get('Hyperparameters', 'SEQ_LENGTH'))
STEP_SIZE = int(config.get('Hyperparameters', 'STEP_SIZE'))
EMBEDDING_DIM = int(config.get('Hyperparameters', 'EMBEDDING_DIM'))
HIDDEN_SIZE = int(config.get('Hyperparameters', 'HIDDEN_SIZE'))
BATCH_SIZE = int(config.get('Hyperparameters', 'BATCH_SIZE'))
NUM_LAYERS = int(config.get('Hyperparameters', 'NUM_LAYERS'))
DECAY_RATE = float(config.get('Hyperparameters', 'DECAY_RATE'))
NUM_EPOCHS = int(config.get('Hyperparameters', 'NUM_EPOCHS'))
LEARNING_RATE = float(config.get('Hyperparameters', 'LEARNING_RATE'))
DECAY_STEP = int(config.get('Hyperparameters', 'DECAY_STEP'))
MIN_EPOCHS = int(config.get('Hyperparameters', 'MIN_EPOCHS'))

INPUT_SIZE = vocab_size
OUTPUT_SIZE = vocab_size

# create model
if model_choice == 'RNN':
    model = mdl.RNN(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
elif model_choice == 'LSTM':
    model = mdl.LSTM(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
elif model_choice == 'GRU':
    model = mdl.GRU(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)


# initialize inputs and targets for train and validation
tr_inputs, tr_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE)
ev_inputs, ev_targets = initialize_seq(fulltext, SEQ_LENGTH, STEP_SIZE, train=False)
tr_dataset = data_config.DemandDataset(torch.tensor(tr_inputs).to(device), torch.tensor(tr_targets).to(device))
ev_dataset = data_config.DemandDataset(torch.tensor(ev_inputs).to(device), torch.tensor(ev_targets).to(device))
tr_dataloader = DataLoader(tr_dataset, shuffle=True, batch_size=BATCH_SIZE)     # training set
ev_dataloader = DataLoader(ev_dataset, shuffle=True, batch_size=BATCH_SIZE)     # validation set
n_train = len(tr_dataloader)
n_eval = len(ev_dataloader)


if TRAIN == 'generate':
    state_dict = torch.load(f'pretrained/{model_choice}.pth')
    model.load_state_dict(state_dict)   # load pre-trained state dict
    model.to(device)    # send it to gpu
    model.eval()    # models are saved in train mode

if TRAIN == 'train':
    start = time.time()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)   # SGD works better than Adam for this task
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)  # lr decay through epochs

    # training function
    epoch_tr_losses, epoch_ev_losses, epoch_perplexities = train_epochs(model, tr_dataloader, ev_dataloader, criterion,
                                                                        optimizer, scheduler, NUM_EPOCHS, MIN_EPOCHS)

    end = time.time()
    training_time = end - start
    save_data(model_choice, epoch_tr_losses, epoch_ev_losses, epoch_perplexities, training_time, SAVE, model)


# generate text (if trained, generates text after training)
sampled_txt = inference(model)
print(f'sampled text from {model_choice}: {sampled_txt}')






