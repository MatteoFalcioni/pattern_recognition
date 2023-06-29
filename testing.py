import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import configparser
import models as mdl
import data_config as dc
from data_config import vocab_size, encode, decode, initialize_seq
from training import train, validate, train_epochs, inference

"""
this file contains the testing of the functions in models.py, training.py and data_config.py
"""

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get test hyperparameters
config = configparser.ConfigParser()
config.read('test_data/test_config.txt')
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

test_model = mdl.RNN(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(test_model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)

# build ad-hoc test dataset: inputs and targets are the same
test_txt = open('test_data/test_corpus.txt', 'rb').read().decode(encoding='utf-8').lower()
test_in, test_targ = initialize_seq(test_txt, SEQ_LENGTH, STEP_SIZE)
test_dataset = dc.DemandDataset(torch.tensor(test_in).to(device), torch.tensor(test_targ).to(device))
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


def test_encode_decode():
    """
    this function tests the functions encode() and decode(), by checking that if we first encode a string and
    then decode it we get the original string back
    """
    test_chars = 'ciao'
    assert (decode(encode(test_chars)) == test_chars)


def test_initialize_seq():
    """
    this function test the correct functioning of initialize_seq() by checking that, when given
    a certain sequence, the inputs list will contain the sequence's 'windows' of size seq_length,
    and the targets list will contain the expected character following that window by a shift of step_size
    """
    test_seq = 'abcde'
    test_inputs, test_targets = initialize_seq(test_seq, 2, 1, perc=1.0)    # seq_length = 2, shift of window = 1
    assert (test_inputs[0] == [1, 2])
    assert (test_targets[0] == 3)
    assert (test_inputs[1] == [2, 3])
    assert (test_targets[1] == 4)
    assert (test_inputs[2] == [3, 4])
    assert (test_targets[2] == 5)


def test_train():
    """
    this function tests the train() function, by checking that when we train on the ad-hoc dataset, which has inputs
    equal to the targets, the training loss tends to zero (the model predicts always the right value)
    """
    test_tr_loss = train(test_model, test_dataloader, optimizer)
    assert (test_tr_loss < 0.01)


def test_validate():
    """
        this function tests the validate() function, by checking that when we validate on the ad-hoc dataset,
        which has inputs equal to the targets, the validation loss tends to zero
        (the model predicts always the right value) and thus the perplexity tends to 1
    """
    test_ev_loss, test_perplexity = validate(test_model, test_dataloader)
    assert (test_ev_loss < 0.01)
    assert (test_perplexity < 1.01)     # perplexity = exp(loss), tends to 1


def test_train_epochs():
    """
        this function tests the train_epochs() function, by checking that when we train and validate on
        the ad-hoc dataset (inputs = targets) over different epochs, all the epoch losses tend to zero
        and the epoch perplexities tend to 1. We test it for 2 epochs
    """
    epoch_tr_losses, epoch_ev_losses, epoch_perplexities = train_epochs(test_model, test_dataloader, test_dataloader,
                                                                        NUM_EPOCHS, MIN_EPOCHS, optimizer, scheduler)
    assert (epoch_tr_losses[0] < 0.01)
    assert (epoch_ev_losses[0] < 0.01)
    assert (epoch_perplexities[0] < 1.01)
    assert (epoch_tr_losses[1] < 0.01)
    assert (epoch_ev_losses[1] < 0.01)
    assert (epoch_perplexities[1] < 1.01)


def test_inference():
    """
        this function tests the inference() function by checking that, when we sample from the model trained on the
        ad-hoc dataset, the model will always infer the same letter (in our case, the letter 'a' since the ad hoc
        dataset contains only a's)
    """
    test_sample = inference(test_model, 'init', 50)
    a_string = 'a' * 50
    assert (test_sample[len('init'):] == a_string)









