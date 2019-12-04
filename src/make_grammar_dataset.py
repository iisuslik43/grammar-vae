from __future__ import print_function
import nltk
import pdb
from src import reactions_grammar
import numpy as np
import h5py
from tqdm import tqdm



MAX_LEN = 300
NCHARS = len(reactions_grammar.GCFG.productions())

def get_reactions_tokenizer(cfg):
    long_tokens = list(filter(lambda a: len(a) > 1, cfg._lexical_index.keys()))
    replacements = ['$', '%', '^', '&']  # ,'&']
    assert len(long_tokens) == len(replacements)
    for token in replacements:
        assert token not in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize


def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(reactions_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = get_reactions_tokenizer(reactions_grammar.GCFG)
    tokens = list(map(tokenize, smiles))
    parser = nltk.ChartParser(reactions_grammar.GCFG)
    parse_trees = [list(parser.parse(t))[0] for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN), -1] = 1.
    return one_hot


if __name__ == '__main__':
    # f = open('grammarVAE/data/250k_rndm_zinc_drugs_clean.smi','r')
    # f = open('data/biocad_dataset.smi','r')
    f = open('data/biocad_reactions_dataset.smi', 'r')
    L = []

    count = -1
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()

    step = 100
    max_len = min(len(L), 1000)
    OH = None
    for i in tqdm(range(0, max_len, step)):
        # print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
        try:
            onehot = to_one_hot(L[i:i + step])
            if OH is None:
                OH = onehot
            else:
                OH = np.concatenate((OH, onehot), axis=0)
        except (ValueError, IndexError, StopIteration) as e:
            print('WTF happened: ' + str(e))
            pass
    print(OH.shape)

    # h5f = h5py.File('grammarVAE/data/zinc_grammar_dataset.h5','w')
    # h5f = h5py.File('data/biocad_grammar_dataset.h5','w')
    h5f = h5py.File('data/biocad_reactions_grammar_dataset.h5', 'w')
    h5f.create_dataset('data', data=OH)
    h5f.close()
