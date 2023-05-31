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
    for f_i in range(3):
        raw_data_dir = '/home/swang/xuhw/research-projects/data/xenograft_data/classification'
        output_data_dir = f'/home/swang/xuhw/research-projects/data/Pisces/xenograft_data/classification/fold{f_i}'

        print("processing start !")
        
        os.makedirs(output_data_dir, exist_ok=True)
        
        drug_smiles_dir = '/home/swang/xuhw/research-projects/data/Pisces/xenograft_data/drug_smiles.txt'
        
        model_idxes_dir = '/home/swang/xuhw/research-projects/data/Pisces/xenograft_data/model_fpkm.csv'
        model_names = pd.read_csv(model_idxes_dir)['model_names']
        model_TO_INDEX_DICT = {model_names[idx]: idx for idx in range(len(model_names))}

        label_field = 'label'

        all_drug_dict = {}
        name_list, smiles_list = [], []
        with open(drug_smiles_dir, 'r') as f:
            for line in f:
                name, smiles = line.strip().split('\t')
                name_list.append(name)
                smiles_list.append(smiles)
        for idx, smiles in zip(name_list, smiles_list):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
            all_drug_dict[idx] = smi_tokenizer(smiles)
        
        drug_id_dict = {}
        id = 0
        for idx in name_list:
            drug_id_dict[idx] = id
            id += 1

        with open(os.path.join(output_data_dir, 'drug_name.dict'), 'w') as f1, \
                open(os.path.join(output_data_dir, 'drug_id.dict'), 'w') as f2:
            for key, value in drug_id_dict.items():
                f1.write(str(key) + ' ' + str(value) + '\n')
                f2.write(str(value) + ' ' + str(value) + '\n')
        f1.close()
        f2.close()
        
        train_pairs, valid_pairs, test_pairs = [], [], []
        train_csv_path = os.path.join(output_data_dir, 'train.csv')
        valid_csv_path = os.path.join(output_data_dir, 'valid.csv')
        test_csv_path = os.path.join(output_data_dir, 'test.csv')

        train_a_dir = os.path.join(output_data_dir, 'train.a')
        train_b_dir = os.path.join(output_data_dir, 'train.b')
        train_model_dir = os.path.join(output_data_dir, 'train.model')
        valid_a_dir = os.path.join(output_data_dir, 'valid.a')
        valid_b_dir = os.path.join(output_data_dir, 'valid.b')
        valid_model_dir = os.path.join(output_data_dir, 'valid.model')
        test_a_dir = os.path.join(output_data_dir, 'test.a')
        test_b_dir = os.path.join(output_data_dir, 'test.b')
        test_model_dir = os.path.join(output_data_dir, 'test.model')

        train_label_dir = os.path.join(output_data_dir, 'train.label')
        valid_label_dir = os.path.join(output_data_dir, 'valid.label')
        test_label_dir = os.path.join(output_data_dir, 'test.label')

        label_dict = {'1': 1, '2': 2, '3': 3, '4': 4}
        model_dict = {}

        model_dict = {str(i): i for i in range(len(model_names))}
        
        drug_unusal = set()

        with open(train_a_dir, 'w') as ta_w, open(train_b_dir, 'w') as tb_w, \
            open(valid_a_dir, 'w') as va_w, open(valid_b_dir, 'w') as vb_w, \
            open(test_a_dir, 'w') as tsa_w, open(test_b_dir, 'w') as tsb_w, \
            open(train_model_dir, 'w') as tc, open(valid_model_dir, 'w') as vc, \
            open(test_model_dir, 'w') as tsc, open(train_label_dir, 'w') as tl, \
            open(valid_label_dir, 'w') as vl, open(test_label_dir, 'w') as tsl:

            train_csv = pd.read_csv(train_csv_path)
            for a, b, model, y in zip(tqdm(train_csv['drug_a']), train_csv['drug_b'], \
                train_csv['model'], train_csv[label_field]):
                if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                    ta_w.writelines(all_drug_dict[a] + '\n')
                    tb_w.writelines(all_drug_dict[b] + '\n')
                    tc.writelines(str(model_TO_INDEX_DICT[model]) + '\n')

                    tl.writelines(str(y) + '\n')

                    # if str(model_TO_INDEX_DICT[model]) not in model_dict:
                    #     model_dict[str(model_TO_INDEX_DICT[model])] = len(model_dict)
                
                if a not in all_drug_dict.keys():
                    drug_unusal.add(a) 
                if b not in all_drug_dict.keys():
                    drug_unusal.add(b)
            
            valid_csv = pd.read_csv(valid_csv_path)
            for a, b, model, y in zip(tqdm(valid_csv['drug_a']), valid_csv['drug_b'], \
                valid_csv['model'], valid_csv[label_field]):
                if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                    va_w.writelines(all_drug_dict[a] + '\n')
                    vb_w.writelines(all_drug_dict[b] + '\n')
                    vc.writelines(str(model_TO_INDEX_DICT[model]) + '\n')

                    vl.writelines(str(y) + '\n')
                    
                    # if str(model_TO_INDEX_DICT[model]) not in model_dict:
                    #     model_dict[str(model_TO_INDEX_DICT[model])] = len(model_dict)

                if a not in all_drug_dict.keys():
                    drug_unusal.add(a) 
                if b not in all_drug_dict.keys():
                    drug_unusal.add(b)

            test_csv = pd.read_csv(test_csv_path)
            for a, b, model, y in zip(tqdm(test_csv['drug_a']), test_csv['drug_b'], \
                test_csv['model'], test_csv[label_field]):
                if a in all_drug_dict.keys() and b in all_drug_dict.keys():
                    tsa_w.writelines(all_drug_dict[a] + '\n')
                    tsb_w.writelines(all_drug_dict[b] + '\n')
                    tsc.writelines(str(model_TO_INDEX_DICT[model]) + '\n')

                    tsl.writelines(str(y) + '\n')

                    # if str(model_TO_INDEX_DICT[model]) not in model_dict:
                    #     model_dict[str(model_TO_INDEX_DICT[model])] = len(model_dict)

                if a not in all_drug_dict.keys():
                    drug_unusal.add(a) 
                if b not in all_drug_dict.keys():
                    drug_unusal.add(b)
        
        label_dict_dir = os.path.join(output_data_dir, 'label.dict')
        with open(label_dict_dir, 'w') as label_dict_w:
            for label_name, label_idx in label_dict.items():
                label_dict_w.writelines(label_name + " " + str(label_idx) + '\n')

        model_dict_dir = os.path.join(output_data_dir, 'model.dict')
        with open(model_dict_dir, 'w') as model_dict_w:
            for model_name, model_idx in model_dict.items():
                model_dict_w.writelines(model_name + " " + str(model_idx) + '\n')
        
        for a, b in zip(tqdm(train_csv['drug_a']), train_csv['drug_b']):
            train_pairs.append((drug_id_dict[a], drug_id_dict[b]))

        for a, b in zip(tqdm(valid_csv['drug_a']), valid_csv['drug_b']):
            valid_pairs.append((drug_id_dict[a], drug_id_dict[b]))

        for a, b in zip(tqdm(test_csv['drug_a']), test_csv['drug_b']):
            test_pairs.append((drug_id_dict[a], drug_id_dict[b]))

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

        
        print(f'drug_unusal:{drug_unusal}')        
        print("processing done !")
        
if __name__ == "__main__":
    main()
