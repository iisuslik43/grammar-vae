import sys
sys.path.append('grammar-vae')
import torch
import torch.nn as nn
import torch.utils.data
from src.make_grammar_dataset import NCHARS, MAX_LEN
from src.model import GrammarVAE
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

class GrammarVAETrainingModel(nn.Module):
    """Grammar Variational Autoencoder"""

    def __init__(self):
        super(GrammarVAETrainingModel, self).__init__()
        self.grammar_vae = GrammarVAE(20, 21, 22, NCHARS, 'gru')

    def forward(self, x):
        mu, sigma = self.grammar_vae.encoder(x)
        kl_loss = self.grammar_vae.kl(mu, sigma)
        logits = self.grammar_vae(x, MAX_LEN)
        return logits, kl_loss


class VaeLoss(nn.Module):
    """Grammar Variational Autoencoder"""

    def __init__(self):
        super(VaeLoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, y, logits, kl_loss):
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)

        reconstruction_loss = self.cross_entropy(logits, y)
        return KL_WEIGHT * kl_loss + reconstruction_loss


def draw_losses(train_losses, test_losses):
    plt.plot(list(range(len(train_losses))), train_losses, color='r', label='train')
    plt.plot(list(range(len(test_losses))), test_losses, color='b', label='test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 100
    OUTPUT_SIZE = NCHARS
    KL_WEIGHT = 1
    N_EPOCHS = 10

    # Load data
    data_path = 'data/biocad_reactions_grammar_dataset.h5'
    f = h5py.File(data_path, 'r')
    data = f['data']
    test_split = int(len(data) * 0.8)
    dataset_train = torch.utils.data.DataLoader(data[:test_split], batch_size=BATCH_SIZE, shuffle=False)
    dataset_test = torch.utils.data.DataLoader(data[test_split:], batch_size=BATCH_SIZE, shuffle=False)

    criterion = VaeLoss()
    model = GrammarVAETrainingModel().to(device)
    optimizer = torch.optim.Adam(params=model.parameters())

    train_losses = []
    test_losses = []
    for epoch in tqdm(range(N_EPOCHS)):
        for X_batch in dataset_train:
            optimizer.zero_grad()
            x = X_batch.transpose(-2, -1).float().to(device)
            _, y = x.max(1)
            logits, kl_loss = model(x)
            loss = criterion(y, logits, kl_loss)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()

            train_losses.append(train_loss)
            with torch.no_grad():
                cur_losses = []
                for test_batch in dataset_test:
                    x_test = test_batch.transpose(-2, -1).float().to(device)
                    _, y_test = x_test.max(1)
                    logits_test, kl_loss_test = model(x_test)
                    test_loss = criterion(y_test, logits_test, kl_loss_test).item()
                    cur_losses.append(test_loss)
                test_losses.append(np.mean(cur_losses))
    draw_losses(train_losses, test_losses)