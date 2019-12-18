import sys
import torch
import h5py
sys.path.append('grammar-vae')
from src.grammar_vae import GrammarVAETrainingModel
from src.reactions_grammar import productions_to_string

if __name__ == '__main__':
    device = torch.device('cpu')
    model = GrammarVAETrainingModel(device)
    model.load_state_dict(torch.load('model/model.pt', map_location='cpu'))

    # Load data
    data_path = 'data/biocad_reactions_grammar_dataset.h5'
    f = h5py.File(data_path, 'r')
    data = f['data']
    x = torch.from_numpy(data[:4]).transpose(-2, -1).float()

    mu, sigma = model.grammar_vae.encoder(x)
    z = model.grammar_vae.sample(mu, sigma)
    rules_all = model.grammar_vae.generate(z, sample=False, max_length=300)
    for rules in rules_all:
        print(rules)
