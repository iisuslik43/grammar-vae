import torch
import torch.nn as nn
from src.make_grammar_dataset import NCHARS, MAX_LEN
from torch.autograd import Variable
from src.model import GrammarVAE
from src.reactions_grammar import productions_to_string
import h5py


if __name__ == '__main__':

    BATCH_SIZE = 100
    OUTPUT_SIZE = NCHARS
    KL_WEIGHT = 1

    # Load data
    data_path = 'data/biocad_reactions_grammar_dataset.h5'
    f = h5py.File(data_path, 'r')
    data = f['data']

    # Create encoder
    model = GrammarVAE(20, 21, 22, NCHARS, 'gru')

    # Pass through some data
    x = torch.from_numpy(data[:BATCH_SIZE]).transpose(-2, -1).float() # shape [batch, 12, 15] - nope
    #x = Variable(x)
    _, y = x.max(1) # The rule index
    mu, sigma = model.encoder(x)
    z = model.sample(mu, sigma)
    kl_loss = model.kl(mu, sigma)
    logits = model(x, MAX_LEN)

    criterion = torch.nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)

    reconstruction_loss = criterion(logits, y)
    loss = KL_WEIGHT * kl_loss + reconstruction_loss
    loss.backward()

    print(loss)

    rules_all = model.generate(z, sample=False, max_length=MAX_LEN)
    for rules in rules_all:
        print(productions_to_string(rules))

