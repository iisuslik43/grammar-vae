import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from nltk import Nonterminal

from src.encoder import Encoder
from src.decoder import Decoder
from src.stack import Stack
from src.reactions_grammar import GCFG, START, get_mask


class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder"""

    def __init__(self, hidden_encoder_size, z_dim, hidden_decoder_size, output_size, rnn_type):
        super(GrammarVAE, self).__init__()
        self.encoder = Encoder(hidden_encoder_size, z_dim)
        self.decoder = Decoder(z_dim, hidden_decoder_size, output_size, rnn_type)

    def sample(self, mu, sigma):
        """Reparametrized sample from a N(mu, sigma) distribution"""
        normal = Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        eps = normal.sample()
        z = mu + eps * torch.sqrt(sigma)
        return z

    def kl(self, mu, sigma):
        """KL divergence between two normal distributions"""
        return torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), 1))

    def forward(self, x, max_length):
        mu, sigma = self.encoder(x)
        z = self.sample(mu, sigma)
        logits = self.decoder(z, max_length=max_length)
        return logits

    def generate(self, z, sample, max_length):
        """Generate a valid expression from z using the decoder"""
        logits_all = self.decoder(z, max_length=max_length).squeeze()
        rules_all = []
        for logits in logits_all:
            t = 0
            stack = Stack(grammar=GCFG, start_symbol=START)
            rules = []
            while stack.nonempty:
                alpha = stack.pop()
                mask = get_mask(alpha, stack.grammar, as_variable=True)
                probs = mask * logits[t].exp()
                probs = probs / probs.sum()
                if sample:
                    m = Categorical(probs)
                    i = m.sample()
                else:
                    _, i = probs.max(-1)  # argmax
                # convert PyTorch Variable to regular integer
                i = i.item()
                # select rule i
                rule = stack.grammar.productions()[i]
                rules.append(rule)
                # add rhs nonterminals to stack in reversed order
                for symbol in reversed(rule.rhs()):
                    if isinstance(symbol, Nonterminal):
                        stack.push(symbol)
                t += 1
                if t == max_length:
                    break
            rules_all.append(rules)
        # if len(rules) < 15:
        #     pad = [stack.grammar.productions()[-1]]

        return rules_all
