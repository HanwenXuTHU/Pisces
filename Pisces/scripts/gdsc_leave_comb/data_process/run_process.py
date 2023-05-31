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
    parser.add_argument('--root_data_dir', type=str, default='/home/swang/xuhw/research-projects/data/Pisces/leave_combs')
    args = parser.parse_args()
    for f_i in range(5):
        print("processing start !")
        
        # add args raw_data_dir, output_data_dir
        raw_data_dir = 'data/drug_combo'
        
        output_data_dir = f'{args.root_data_dir}/fold{f_i}'
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
        
        train_csv_path = os.path.join(output_data_dir, 'train.csv')
        valid_csv_path = os.path.join(output_data_dir, 'valid.csv')
        test_csv_path = os.path.join(output_data_dir, 'test.csv')

        train_a_dir = os.path.join(output_data_dir, 'train.a')
        train_b_dir = os.path.join(output_data_dir, 'train.b')
        train_cell_dir = os.path.join(output_data_dir, 'train.cell')
        valid_a_dir = os.path.join(output_data_dir, 'valid.a')
        valid_b_dir = os.path.join(output_data_dir, 'valid.b')
        valid_cell_dir = os.path.join(output_data_dir, 'valid.cell')
        test_a_dir = os.path.join(output_data_dir, 'test.a')
        test_b_dir = os.path.join(output_data_dir, 'test.b')
        test_cell_dir = os.path.join(output_data_dir, 'test.cell')

        train_label_dir = os.path.join(output_data_dir, 'train.label')
        valid_label_dir = os.path.join(output_data_dir, 'valid.label')
        test_label_dir = os.path.join(output_data_dir, 'test.label')

        label_dict = {'0': 0, '1': 1}
        cell_dict = {}

        cell_dict = {str(i): i for i in range(len(cell_names))}

        drug_unusal = set()

        with open(train_a_dir, 'w') as ta_w, open(train_b_dir, 'w') as tb_w, \
            open(valid_a_dir, 'w') as va_w, open(valid_b_dir, 'w') as vb_w, \
            open(test_a_dir, 'w') as tsa_w, open(test_b_dir, 'w') as tsb_w, \
            open(train_cell_dir, 'w') as tc, open(valid_cell_dir, 'w') as vc, \
            open(test_cell_dir, 'w') as tsc, open(train_label_dir, 'w') as tl, \
            open(valid_label_dir, 'w') as vl, open(test_label_dir, 'w') as tsl:

            train_csv = pd.read_csv(train_csv_path)
            for a, b, cell, y in zip(tqdm(train_csv['anchor_names']), train_csv['library_names'], \
                train_csv['cell_line_names'], train_csv['labels']):
                if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                    ta_w.writelines(all_drug_dict[a] + '\n')
                    tb_w.writelines(all_drug_dict[b] + '\n')
                    tc.writelines(str(CELL_TO_INDEX_DICT[cell]) + '\n')

                    tl.writelines(str(y) + '\n')
                
                if a not in all_drug_dict.keys():
                    drug_unusal.add(a) 
                if b not in all_drug_dict.keys():
                    drug_unusal.add(b)
            
            valid_csv = pd.read_csv(valid_csv_path)
            for a, b, cell, y in zip(tqdm(valid_csv['anchor_names']), valid_csv['library_names'], \
                valid_csv['cell_line_names'], valid_csv['labels']):
                if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                    va_w.writelines(all_drug_dict[a] + '\n')
                    vb_w.writelines(all_drug_dict[b] + '\n')
                    vc.writelines(str(CELL_TO_INDEX_DICT[cell]) + '\n')

                    vl.writelines(str(y) + '\n')

                if a not in all_drug_dict.keys():
                    drug_unusal.add(a) 
                if b not in all_drug_dict.keys():
                    drug_unusal.add(b)

            test_csv = pd.read_csv(test_csv_path)
            for a, b, cell, y in zip(tqdm(test_csv['anchor_names']), test_csv['library_names'], \
                test_csv['cell_line_names'], test_csv['labels']):
                if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                    tsa_w.writelines(all_drug_dict[a] + '\n')
                    tsb_w.writelines(all_drug_dict[b] + '\n')
                    tsc.writelines(str(CELL_TO_INDEX_DICT[cell]) + '\n')

                    tsl.writelines(str(y) + '\n')

                if a not in all_drug_dict.keys():
                    drug_unusal.add(a) 
                if b not in all_drug_dict.keys():
                    drug_unusal.add(b)
        

        train_pairs, valid_pairs, test_pairs = [], [], []
        for a, b in zip(tqdm(train_csv['anchor_names']), train_csv['library_names']):
            train_pairs.append((drug2id[a], drug2id[b]))

        for a, b in zip(tqdm(valid_csv['anchor_names']), valid_csv['library_names']):
            valid_pairs.append((drug2id[a], drug2id[b]))

        for a, b in zip(tqdm(test_csv['anchor_names']), test_csv['library_names']):
            test_pairs.append((drug2id[a], drug2id[b]))

        with open(os.path.join(output_data_dir, 'train.pair'), 'w') as tr_w:
            for a, b in train_pairs:
                tr_w.write(str(a) + ' ' + str(b) + '\n')
        tr_w.close()

        with open(os.path.join(output_data_dir, 'valid.pair'), 'w') as va_w:
            for a, b in valid_pairs:
                va_w.write(str(a) + ' ' + str(b) + '\n')
        va_w.close()

        with open(os.path.join(output_data_dir, 'test.pair'), 'w') as te_w:
            for a, b in test_pairs:
                te_w.write(str(a) + ' ' + str(b) + '\n')
        te_w.close()

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
