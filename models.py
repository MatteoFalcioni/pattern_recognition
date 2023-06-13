import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


class RNN(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.system = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer. Change it to nonlinear

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        out, hidden_state = self.system(x_emb)

        if x.size(0) > 1:
            out = out[:, -1, :]

        out = self.fc(out)
        return out, hidden_state

    def sample(self, seed):     # seed can either be a char or a sequence
        self.eval()
        with torch.no_grad():
            seed = self.embedding(torch.tensor(encode(seed)))
            output, _ = self.system(seed)
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


class LSTM(RNN, nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_size, num_layers):
        super(LSTM, self).__init__(input_size, output_size, embedding_dim, hidden_size, num_layers)
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.system = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer


class GRU(RNN, nn.Module):
    def __init__(self, input_size, output_size, embedding_dim,  hidden_size, num_layers):
        super(GRU, self).__init__(input_size, output_size, embedding_dim, hidden_size, num_layers)
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.system = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer


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



