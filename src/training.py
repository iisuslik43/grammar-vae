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
from src.util import Timer, AnnealKL
import wandb



class GrammarVAETrainingModel(nn.Module):
    """Grammar Variational Autoencoder"""

    def __init__(self, device):
        super(GrammarVAETrainingModel, self).__init__()
        self.grammar_vae = GrammarVAE(20, 21, 22, NCHARS, 'gru', device)
        self.device = device

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
        self.kl_weight = 1
        self.anneal = AnnealKL(step=1e-3, rate=500)

    def forward(self, y, logits, kl_loss):
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        reconstruction_loss = self.cross_entropy(logits, y)
        return self.kl_weight * kl_loss + reconstruction_loss


def draw_losses(train_losses, test_losses):
    plt.plot(list(range(len(train_losses))), train_losses, color='r', label='train')
    plt.plot(list(range(len(test_losses))), test_losses, color='b', label='test')
    plt.legend()
    plt.show()


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 256
    N_EPOCHS = 20
    wandb.init(project="grammar-vae")

    # Load data
    data_path = 'data/biocad_reactions_grammar_dataset.h5'
    f = h5py.File(data_path, 'r')
    data = f['data']
    test_split = int(len(data) * 0.8)
    dataset_train = torch.utils.data.DataLoader(data[:test_split], batch_size=BATCH_SIZE, shuffle=False)
    dataset_test = torch.utils.data.DataLoader(data[test_split:], batch_size=BATCH_SIZE, shuffle=False)

    criterion = VaeLoss()
    model = GrammarVAETrainingModel(device).to(device)
    optimizer = torch.optim.Adam(params=model.parameters())

    wandb.watch(model)

    for epoch in tqdm(range(1, N_EPOCHS + 1)):
        for step, X_batch in enumerate(dataset_train, 1):
            optimizer.zero_grad()
            x = X_batch.transpose(-2, -1).float().to(device)
            _, y = x.max(1)
            logits, kl_loss = model(x)

            #criterion.kl_weight = criterion.anneal.alpha(step)
            loss = criterion(y, logits, kl_loss)

            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            with torch.no_grad():
                reconstruction_loss = criterion.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                for test_batch in dataset_test:
                    x_test = test_batch.transpose(-2, -1).float().to(device)
                    _, y_test = x_test.max(1)
                    logits_test, kl_loss_test = model(x_test)
                    test_loss = criterion(y_test, logits_test, kl_loss_test).item()
                    reconstruction_loss_test = criterion.cross_entropy(logits_test.view(-1, logits.size(-1)), y_test.view(-1))
                    break
                test_loss = np.mean(test_loss)
            wandb.log({"Test Loss": test_loss,
                       "Test KL": kl_loss_test,
                       "Test Reconstruction": reconstruction_loss_test,
                       "Train Loss": train_loss,
                       "Train KL": kl_loss,
                       "Train Reconstruction": reconstruction_loss
                       })
        if epoch % 10 == 0:
            torch.save(model, f'model/model_{epoch}.pt')
    torch.save(model, f'model/model.pt')
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    train()
