import sys
import torch
sys.path.append('grammar-vae')
from src.grammar_vae import GrammarVAETrainingModel

if __name__ == '__main__':
    device = torch.device('cpu')
    model = GrammarVAETrainingModel(device)
    model.load_state_dict(torch.load('model/model.pt', map_location='cpu'))