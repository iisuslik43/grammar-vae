import torch
import torch.nn as nn


class Decoder(nn.Module):
    """RNN decoder that reconstructs the sequence of rules from laten z"""

    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError('Wrong rnn_type, should be from [lstm, gru]')

        self.relu = nn.ReLU()

    def forward(self, z, max_length):
        """The forward pass used for training the Grammar VAE.

        For the rnn we follow the same convention as the official keras
        implementaion: the latent z is the input to the rnn at each timestep.
        See line 138 of
            https://github.com/mkusner/grammarVAE/blob/master/models/model_eq.py
        for reference.
        """
        x = self.linear_in(z)
        x = self.relu(x)
        # The input to the rnn is the same for each timestep: it is z.
        x = x.unsqueeze(1).expand(-1, max_length, -1)
        hx = torch.zeros(x.size(0), self.hidden_size).reshape(1, x.size(0), self.hidden_size)
        hx = (hx, hx) if self.rnn_type == 'lstm' else hx
        x, _ = self.rnn(x, hx)

        x = self.relu(x)
        x = self.linear_out(x)
        return x
