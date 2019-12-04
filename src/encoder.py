import torch
import torch.nn as nn
import h5py
from torch.autograd import Variable
from src.make_grammar_dataset import NCHARS, MAX_LEN


class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE.

    Applies a series of one-dimensional convolutions to a batch
    of one-hot encodings of the sequence of rules that generate
    an artithmetic expression.
    """

    def __init__(self, hidden_dim=20, z_dim=2, conv_size='small', rules_count=NCHARS):
        super(Encoder, self).__init__()
        if conv_size == 'small':
            # 12 rules, so 12 input channels
            self.conv1 = nn.Conv1d(rules_count, 2, kernel_size=2)
            self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
            self.conv3 = nn.Conv1d(3, 4, kernel_size=4)
            self.linear = nn.Linear(1176, hidden_dim)
        elif conv_size == 'large':
            self.conv1 = nn.Conv1d(rules_count, 24, kernel_size=2)
            self.conv2 = nn.Conv1d(24, 12, kernel_size=3)
            self.conv3 = nn.Conv1d(12, 12, kernel_size=4)
            self.linear = nn.Linear(3528, hidden_dim)
        else:
            raise ValueError('Wron vaglue for conv_size, it should be in [small, large]')

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Encode x into a mean and variance of a Normal"""
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.relu(h)
        h = h.view(x.size(0), -1)  # flatten
        h = self.linear(h)
        h = self.relu(h)
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))
        return mu, sigma

