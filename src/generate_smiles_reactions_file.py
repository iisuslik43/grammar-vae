import json
from tqdm import tqdm
from rdkit import Chem

if __name__ == '__main__':
    data_json = []
    for line in open('data/reactions_02_10_2019.json'):
        line_json = json.loads(line)
        data_json.append(line_json)
    print('Reactions count:', len(data_json))

    with open('data/biocad_reactions_dataset.smi', 'w') as f:
        res = []
        for reaction in tqdm(data_json):
            products = reaction['products']
            reactants = reaction['reactants']
            if len(products) != 1 and len(reactants) != 1:
                continue
            reactant = Chem.MolToSmiles(Chem.MolFromSmiles(reactants[0]))
            product = Chem.MolToSmiles(Chem.MolFromSmiles(products[0]))
            res.append(reactant + '>>' + product)
        f.write('\n'.join(res))