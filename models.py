import torch
import torch.nn as nn
import torch.nn.functional as F
from data_config import encode, ix_to_char

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# fix the seed for reproducibility
torch.manual_seed(1234567890)

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
# for comparison: BATCH_SIZE = N, SEQ_LENGTH = L, D=1 (directions), Hin = EMBEDDING_DIM, Hout = HIDDEN_SIZE


class RNN(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)   # each char of seq is embedded
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.system = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        # if batch_first = True : x has to be (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, output_size)  # linear layer.

    def forward(self, x):
        """
        this method implements the forward process of the RNN module.

        Parameters
            x : input tensor of dimension (N, L, Hin) for batched input or (L, Hin) for unbatched input

        Returns:
            the resulting logits and the last hidden state
        """

        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        # input shape: (N, L, Hin) or (L, Hin) for unbatched input
        out, hidden_state = self.system(x_emb)
        # out is (batch_size, seq_length, hidden_size) or (seq_length, hidden_size) for unbatched

        # out: (N, L, Hout). But we don't need all the chars of the sequence, just the last one
        if x.size(0) > 1:   # x.size(0) == batch_size >1 <--> batched input
            out = out[:, -1, :]

        out = self.fc(out)
        return out, hidden_state

    def sample(self, seed):
        """This method forwards a seed sequence to the model and samples from the output probability distribution
            Parameters
                seed : input sequence (as string) to feed to the model

            Returns:
                the sampled index from the resulting probability distribution obtained from the model
        """

        self.eval()
        with torch.no_grad():
            seed = self.embedding(torch.tensor(encode(seed)))
            output, _ = self.system(seed)   # no layer normalization in sampling
            output = output[-1, :]   # select last char probabilities
            logits = self.fc(output)
            prob = F.softmax(logits, dim=0)
            sample_ix = torch.multinomial(prob, 1, replacement=True).item()
            return ix_to_char[sample_ix]


class LSTM(RNN, nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_size, num_layers):
        super(LSTM, self).__init__(input_size, output_size, embedding_dim, hidden_size, num_layers)
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.system = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


class GRU(RNN, nn.Module):
    def __init__(self, input_size, output_size, embedding_dim,  hidden_size, num_layers):
        super(GRU, self).__init__(input_size, output_size, embedding_dim, hidden_size, num_layers)
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.system = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)



