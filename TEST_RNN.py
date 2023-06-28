import sys
import torch
import configparser

import models as mdl
from data_config import vocab_size, char_to_ix, ix_to_char, encode, decode, initialize_seq

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
model = mdl.RNN(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)


def test_encode_decode():
    """
    this function tests the functions encode() and decode(), by checking that the application of one first and the other
    after results back to the initial input
    """
    test_chars = 'ciao'
    assert (decode(encode(test_chars)) == test_chars)


def test_initialize_seq():
    """
    this function test the correct functioning of initialize_seq(),
    by checking that, when given a certain sequence, the input list and the target list contain respectevely
    the sequence 'windows' of chosen length and the expected character following that window
    """
    test_seq = 'abcde'
    test_inputs, test_targets = initialize_seq(test_seq, 2, 1, perc=1.0)
    assert (test_inputs[0] == [1, 2])
    assert (test_targets[0] == 3)
    assert (test_inputs[1] == [2, 3])
    assert (test_targets[1] == 4)
    assert (test_inputs[2] == [3, 4])
    assert (test_targets[2] == 5)


# create an ad-hoc dataset so that train loss is always zero. i.e., make the inputs and targets always equal. maybe a
# sequence of 25 letters 'a' and targets which are all 'a'. then check that the loss is zero.
# can do the same thing with validation.

# after training on this dataset you could use it also to test sample(), i.e. it should sample always 'a'.

def test_train():
    """
    this function tests the train() function, by checking that ...
    """









