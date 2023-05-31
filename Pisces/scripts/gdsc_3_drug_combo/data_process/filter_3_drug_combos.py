import os
from tqdm import tqdm
from rdkit import Chem
import pandas as pd


def main():
    path = '/home/swang/xuhw/research-projects/data/Pisces/gdsc_3_drug_combo/3-drug-combos.csv'
    df = pd.read_csv(path, index_col=0)
    
    raw_data_dir = 'data/drug_combo'
    drug_smiles_dir = os.path.join(raw_data_dir, 'drug_smiles.csv')
    cell_idxes_dir = os.path.join(raw_data_dir, 'cell_tpm.csv')
    cell_names = pd.read_csv(cell_idxes_dir, index_col=0)['cell_line_names']
    CELL_TO_INDEX_DICT = {cell_names[idx]: idx for idx in range(len(cell_names))}
    cell_set = set(CELL_TO_INDEX_DICT.keys())
    all_drug_dict = {}
    df_drug_smiles = pd.read_csv(drug_smiles_dir, index_col=0)
    drug_set = df_drug_smiles['drug_names'].tolist()

    known_cell = set()
    unknown_cell = set()
    drop_indices = []
    for i in df.index:
        # remove the rows with cell line and drugs not in the set
        known_cell.add(df.loc[i, 'cell_line_names'])
        if df.loc[i, 'cell_line_names'] not in cell_set \
            or df.loc[i, 'anchor_names_1'] not in drug_set \
            or df.loc[i, 'anchor_names_1'] not in drug_set \
            or df.loc[i, 'library_names'] not in drug_set:
            drop_indices.append(i)
        if df.loc[i, 'cell_line_names'] not in cell_set:
            unknown_cell.add(df.loc[i, 'cell_line_names'])
    df = df.drop(drop_indices)
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(raw_data_dir, '/home/swang/xuhw/research-projects/data/Pisces/gdsc_3_drug_combo/3-drug-combos-filtered.csv'))
    debug = 0


if __name__ == '__main__':
    main()