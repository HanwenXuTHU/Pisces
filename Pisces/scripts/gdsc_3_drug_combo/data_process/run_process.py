import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np
import argparse
import pdb

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub('\s+', '', smi) == ''.join(tokens)
    except:
        return ''
    return ' '.join(tokens)

def clean_smiles(smiles):
    t = re.sub(':\d*', '', smiles)
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_dir', type=str, default='/home/swang/xuhw/research-projects/data/Pisces/gdsc_3_drug_combo')
    args = parser.parse_args()

    print("processing start !")
    
    # add args raw_data_dir, output_data_dir
    raw_data_dir = 'data/drug_combo'
    
    output_data_dir = f'{args.root_data_dir}/'
    os.makedirs(output_data_dir, exist_ok=True)
    
    drug_smiles_dir = os.path.join(raw_data_dir, 'drug_smiles.csv')
    
    cell_idxes_dir = os.path.join(raw_data_dir, 'cell_tpm.csv')
    cell_names = pd.read_csv(cell_idxes_dir, index_col=0)['cell_line_names']
    CELL_TO_INDEX_DICT = {cell_names[idx]: idx for idx in range(len(cell_names))}


    all_drug_dict = {}
    df_drug_smiles = pd.read_csv(drug_smiles_dir, index_col=0)
    for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
        all_drug_dict[idx] = smi_tokenizer(smiles)

    drug2id = {}
    id = 0
    for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
        drug2id[idx] = id
        id += 1
    id2drug = {v: k for k, v in drug2id.items()}

    with open(os.path.join(output_data_dir, 'drug_name.dict'), 'w') as f1, \
            open(os.path.join(output_data_dir, 'drug_id.dict'), 'w') as f2:
        for i in range(len(drug2id.keys())):
            f1.write(str(id2drug[i]) + ' ' + str(i) + '\n')
            f2.write(str(i) + ' ' + str(i) + '\n')
    f1.close()
    f2.close()
    
    train_csv_path = os.path.join(output_data_dir, '3-drug-combos-filtered.csv')
    valid_csv_path = os.path.join(output_data_dir, '3-drug-combos-filtered.csv')

    train_a_dir = os.path.join(output_data_dir, 'train.a')
    train_b_dir = os.path.join(output_data_dir, 'train.b')
    train_c_dir = os.path.join(output_data_dir, 'train.c')
    train_cell_dir = os.path.join(output_data_dir, 'train.cell')
    valid_a_dir = os.path.join(output_data_dir, 'valid.a')
    valid_b_dir = os.path.join(output_data_dir, 'valid.b')
    valid_c_dir = os.path.join(output_data_dir, 'valid.c')
    valid_cell_dir = os.path.join(output_data_dir, 'valid.cell')

    train_label_dir = os.path.join(output_data_dir, 'train.label')
    valid_label_dir = os.path.join(output_data_dir, 'valid.label')

    label_dict = {'0': 0, '1': 1}
    cell_dict = {}

    cell_dict = {str(i): i for i in range(len(cell_names))}

    drug_unusal = set()

    with open(train_a_dir, 'w') as ta_w, open(train_b_dir, 'w') as tb_w, open(train_c_dir, 'w') as tc_w, \
        open(valid_a_dir, 'w') as va_w, open(valid_b_dir, 'w') as vb_w, open(valid_c_dir, 'w') as vc_w, \
        open(train_cell_dir, 'w') as tc, open(valid_cell_dir, 'w') as vc, \
        open(train_label_dir, 'w') as tl, open(valid_label_dir, 'w') as vl:

        train_csv = pd.read_csv(train_csv_path)
        for a, b, c, cell, y in zip(tqdm(train_csv['anchor_names_1']), train_csv['anchor_names_B'], train_csv['library_names'], \
            train_csv['cell_line_names'], train_csv['labels']):
            if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                ta_w.writelines(all_drug_dict[a] + '\n')
                tb_w.writelines(all_drug_dict[b] + '\n')
                tc_w.writelines(all_drug_dict[c] + '\n')
                tc.writelines(str(CELL_TO_INDEX_DICT[cell]) + '\n')

                tl.writelines(str(y) + '\n')
            
            if a not in all_drug_dict.keys():
                drug_unusal.add(a) 
            if b not in all_drug_dict.keys():
                drug_unusal.add(b)
        
        valid_csv = pd.read_csv(valid_csv_path)
        for a, b, c, cell, y in zip(tqdm(valid_csv['anchor_names_1']), valid_csv['anchor_names_B'], valid_csv['library_names'], \
            valid_csv['cell_line_names'], valid_csv['labels']):
            if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                va_w.writelines(all_drug_dict[a] + '\n')
                vb_w.writelines(all_drug_dict[b] + '\n')
                vc_w.writelines(all_drug_dict[c] + '\n')
                vc.writelines(str(CELL_TO_INDEX_DICT[cell]) + '\n')

                vl.writelines(str(y) + '\n')

            if a not in all_drug_dict.keys():
                drug_unusal.add(a) 
            if b not in all_drug_dict.keys():
                drug_unusal.add(b)
    

    train_pairs_ab, valid_pairs_ab = [], []
    train_pairs_ac, valid_pairs_ac = [], []
    train_pairs_bc, valid_pairs_bc = [], []
    for a, b in zip(tqdm(train_csv['anchor_names_1']), train_csv['anchor_names_B']):
        train_pairs_ab.append((drug2id[a], drug2id[b]))
    for a, b in zip(tqdm(valid_csv['anchor_names_1']), valid_csv['anchor_names_B']):
        valid_pairs_ab.append((drug2id[a], drug2id[b]))
    for a, c in zip(tqdm(train_csv['anchor_names_1']), train_csv['library_names']):
        train_pairs_ac.append((drug2id[a], drug2id[c]))
    for a, c in zip(tqdm(valid_csv['anchor_names_1']), valid_csv['library_names']):
        valid_pairs_ac.append((drug2id[a], drug2id[c]))
    for b, c in zip(tqdm(train_csv['anchor_names_B']), train_csv['library_names']):
        train_pairs_bc.append((drug2id[b], drug2id[c]))
    for b, c in zip(tqdm(valid_csv['anchor_names_B']), valid_csv['library_names']):
        valid_pairs_bc.append((drug2id[b], drug2id[c]))

    with open(os.path.join(output_data_dir, 'train.pairab'), 'w') as tr_w:
        for a, b in train_pairs_ab:
            tr_w.write(str(a) + ' ' + str(b) + '\n')
    tr_w.close()
    with open(os.path.join(output_data_dir, 'valid.pairab'), 'w') as va_w:
        for a, b in valid_pairs_ab:
            va_w.write(str(a) + ' ' + str(b) + '\n')
    va_w.close()
    with open(os.path.join(output_data_dir, 'train.pairac'), 'w') as tr_w:
        for a, c in train_pairs_ac:
            tr_w.write(str(a) + ' ' + str(c) + '\n')
    tr_w.close()
    with open(os.path.join(output_data_dir, 'valid.pairac'), 'w') as va_w:
        for a, c in valid_pairs_ac:
            va_w.write(str(a) + ' ' + str(c) + '\n')
    va_w.close()
    with open(os.path.join(output_data_dir, 'train.pairbc'), 'w') as tr_w:
        for b, c in train_pairs_bc:
            tr_w.write(str(b) + ' ' + str(c) + '\n')
    tr_w.close()
    with open(os.path.join(output_data_dir, 'valid.pairbc'), 'w') as va_w:
        for b, c in valid_pairs_bc:
            va_w.write(str(b) + ' ' + str(c) + '\n')
    va_w.close()


    label_dict_dir = os.path.join(output_data_dir, 'label.dict')
    with open(label_dict_dir, 'w') as label_dict_w:
        label_dict_w.writelines('0' + " " + str(label_dict['0']) + '\n')
        label_dict_w.writelines('1' + " " + str(label_dict['1']) + '\n')

    cell_dict_dir = os.path.join(output_data_dir, 'cell.dict')
    with open(cell_dict_dir, 'w') as cell_dict_w:
        for i in range(len(cell_dict.keys())):
            cell_dict_w.writelines(str(i) + " " + str(i) + '\n')
    
    print(f'drug_unusal:{drug_unusal}')        
    print("processing done !")
    
if __name__ == "__main__":
    main()
