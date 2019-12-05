import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from src.make_grammar_dataset import NCHARS, MAX_LEN
from src.model import GrammarVAE


class GrammarVAETrainingModel(nn.Module):
    """Grammar Variational Autoencoder"""

    def __init__(self, device):
        super(GrammarVAETrainingModel, self).__init__()
        self.grammar_vae = GrammarVAE(50, 50, 50, NCHARS, 'gru', device)
        self.device = device

    def forward(self, x):
        mu, sigma = self.grammar_vae.encoder(x)
        kl_loss = self.grammar_vae.kl(mu, sigma)
        logits = self.grammar_vae(x, MAX_LEN)
        return logits, kl_loss

    def save(self, filename='model/model.pt'):
        torch.save(self.state_dict(), filename)

    @staticmethod
    def load(device, filename='model/model.pt'):
        model = GrammarVAETrainingModel(device)
        model.load_state_dict(torch.load(filename))
        return model

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)
        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len=300):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)
        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            z = self.sample_z_prior(n_batch).to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                   device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]


