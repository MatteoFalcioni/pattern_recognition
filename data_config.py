import torch
from torch.utils.data import Dataset
import argparse

"""
this file contains helper functions and classes to config data  
"""

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix the seed for reproducibility
torch.manual_seed(1234567890)


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


class DemandDataset(Dataset):
    """
        A class that inherits from the torch Dataset structure to implement custom Dataset

        Attributes
        ----------
        X_train: torch tensor representing the inputs of our neural network
        y_train: torch tensor representing the targets of our inputs

        Methods
        -------
        len()
            returns the length of the dataset

        getitem()
            overrides the torch.Dataset getitem() function to get indexed input and target
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train  # inputs
        self.y_train = y_train  # targets

    def __len__(self):
        """
        This method returns the length of the dataset
        """
        return len(self.y_train)

    def __getitem__(self, idx):
        """
        This method overrides the getitem() function
            Parameters:
                idx: index of the item we want to get
            Returns:
                  the tuple (item corresponding to the index from data, item corresponding to the index from labels)
        """
        data = self.X_train[idx]
        labels = self.y_train[idx]
        return data, labels


def get_parser():
    """
    Returns parser for choosing:
    - the configuration file
    - what model to use
    - whether to train the model or generate from it
    - whether to save or discard the trained parameters
    """
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('-c', '--config', help='configuration file path', default='configuration/config_RNN.txt',
                        type=str)
    parser.add_argument('-m', '--model', choices=['RNN', 'LSTM', 'GRU'], help='Choose the model', default='RNN',
                        type=str)
    parser.add_argument('-t', '--training', choices=['train', 'generate'], help='Choose if the model is going to be '
                                                                                'trained or if it is just generating '
                                                                                'text', default='generate',
                        type=str)
    parser.add_argument('-s', '--saving', choices=['save', 'discard'], help='Choose if the trained model needs to be '
                                                                            'saved', default='discard',
                        type=str)
    return parser


def initialize_seq(corpus, seq_length, step_size, perc=0.8, train=True):
    """
    this function initializes inputs and targets sequences
        Parameters:
            corpus: the pre-processed raw text we want to split in training and validation
            seq_length: the selected length of the sequences
            step_size: the shift of the sequences at each step
            perc: percentage of text to use for training
            train: boolean value to choose whether we are producing training seqs or validation seqs
        Returns:
            the tuple (input sequences, targets)
    """

    encoded_text = encode(corpus)
    k = int(perc * len(corpus))     # train: k% of text --> validation: (100-k)% of text
    if train:
        text = encoded_text[:k]
    else:
        text = encoded_text[k:]

    inputs = []     # input sequences
    targets = []    # target characters for each input seq

    text_len = len(text)
    for i in range(0, text_len - seq_length, step_size):
        inputs.append(text[i: i+seq_length])
        targets.append(text[i + seq_length])

    return inputs, targets



