import torch
import torch.nn.functional as F
import configparser

import models
import data_config
from data_config import encode, char_to_ix, initialize_seq

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix the seed for reproducibility
torch.manual_seed(1234567890)

config = configparser.ConfigParser()
config.read('configuration.txt')

model_choice = 'RNN'

# hyperparameters
SEQ_LENGTH = int(config.get('Hyperparameters', 'SEQ_LENGTH'))
STEP_SIZE = int(config.get('Hyperparameters', 'STEP_SIZE'))
EMBEDDING_DIM = int(config.get('Hyperparameters', 'EMBEDDING_DIM'))
HIDDEN_SIZE = int(config.get('Hyperparameters', 'HIDDEN_SIZE'))
BATCH_SIZE = int(config.get('Hyperparameters', 'BATCH_SIZE'))
NUM_LAYERS = int(config.get('Hyperparameters', 'NUM_LAYERS'))

vocab_size = data_config.vocab_size
INPUT_SIZE = vocab_size
OUTPUT_SIZE = vocab_size
fulltext = data_config.fulltext

model = models.RNN(INPUT_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)


def test_initialize_seq():
    test = fulltext[:200]
    test_enc = encode(fulltext[:200])
    test_input, test_target = initialize_seq(test, SEQ_LENGTH, STEP_SIZE)

    for i in range(0, len(test) - SEQ_LENGTH, STEP_SIZE):
        for k in range(SEQ_LENGTH):
            assert(test_enc[i: i + k] == test_input[i][i: i + k])
        assert(test_enc[i + SEQ_LENGTH] == test_target[i])


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






