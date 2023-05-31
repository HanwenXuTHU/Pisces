import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np
import pdb
import collections


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
    idx_T = 100
    print("processing start !")

    data_dir = f"/mnt/hanoverdev/scratch/hanwen/data/Pisces/ddi1000/"
    save_dir = f"/mnt/hanoverdev/scratch/hanwen/data/Pisces/ddi1000/"
    drug_smiles_dir = "{}{}".format(data_dir, 'drug_smiles.txt')
    
    identified_y2ddi_type = collections.OrderedDict()
    identified_ddi_path = '/home/swang/xuhw/research-projects/Pisces/Pisces/scripts/case_study/ddi_network/output/identified_ddi.txt'
    idx = 0
    for line in open(identified_ddi_path, 'r'):
        y, ddi_type, _, _ = line.strip().split(',')
        ddi_type = eval(ddi_type.split('type ')[-1])
        identified_y2ddi_type[idx] = ddi_type - 1
        idx += 1

    all_drug_dict = {}
    f = open(os.path.join(save_dir, 'drug_smiles.txt'), 'r')
    for line in f:
        d_name, smiles = line.strip().split('\t')
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
        all_drug_dict[d_name] = smi_tokenizer(smiles)

    drug2id = {}
    id = 0
    f = open(os.path.join(save_dir, 'drug_smiles.txt'), 'r')
    for line in f:
        idx, sms = line.strip().split('\t')
        drug2id[idx] = id
        id += 1
    id2drug = {v: k for k, v in drug2id.items()}

    with open(os.path.join(save_dir, 'drug_name.dict'), 'w') as f1, \
            open(os.path.join(save_dir, 'drug_id.dict'), 'w') as f2:
        for i in range(len(drug2id.keys())):
            f1.write(str(id2drug[i]) + ' ' + str(i) + '\n')
            f2.write(str(i) + ' ' + str(i) + '\n')
    f1.close()
    f2.close()
    
    train_csv_file = "{}{}".format(data_dir, 'data.csv')
    valid_csv_file = "{}{}".format(data_dir, 'data.csv')
    
    train_a_dir = "{}{}".format(save_dir, 'train.a')
    train_b_dir = "{}{}".format(save_dir, 'train.b')
    train_pair_dir = "{}{}".format(save_dir, 'train.pair')
    valid_a_dir = "{}{}".format(save_dir, 'valid.a')
    valid_b_dir = "{}{}".format(save_dir, 'valid.b')
    valid_pair_dir = "{}{}".format(save_dir, 'valid.pair')

    train_label_dir = "{}{}".format(save_dir, 'train.label')
    valid_label_dir = "{}{}".format(save_dir, 'valid.label')
    
    with open(train_a_dir, 'w') as ta_w, \
        open(train_b_dir, 'w') as tb_w, \
        open(train_pair_dir, 'w') as tp_w, \
        open(valid_a_dir, 'w') as va_w, \
        open(valid_b_dir, 'w') as vb_w, \
        open(valid_pair_dir, 'w') as vp_w, \
        open(train_label_dir, 'w') as tl, open(valid_label_dir, 'w') as vl:
        
        train_csv = pd.read_csv(train_csv_file, sep='\t')

        for a, b, y in zip(tqdm(train_csv['Drug1_ID']), train_csv['Drug2_ID'], train_csv['Y']):
            
            ta_w.writelines(all_drug_dict[a] + '\n')
            tb_w.writelines(all_drug_dict[b] + '\n')
            tp_w.writelines(str(drug2id[a]) + '\t' + str(drug2id[b]) + '\n')

            # pdb.set_trace()
            tl.writelines(str(identified_y2ddi_type[y]) + '\n')
    
        valid_csv = pd.read_csv(valid_csv_file, sep='\t')

        for a, b, y in zip(tqdm(valid_csv['Drug1_ID']), valid_csv['Drug2_ID'], valid_csv['Y']):
            
            va_w.writelines(all_drug_dict[a] + '\n')
            vb_w.writelines(all_drug_dict[b] + '\n')
            vp_w.writelines(str(drug2id[a]) + '\t' + str(drug2id[b]) + '\n')

            vl.writelines(str(identified_y2ddi_type[y]) + '\n')
    label_dict = list(np.arange(86))
    label_dict_dir  = "{}{}".format(save_dir, 'label.dict')
    label_dict = np.sort(np.asarray(label_dict))
    with open(label_dict_dir, 'w') as label_dict_w:
        for label_name in label_dict:
            label_dict_w.writelines(str(label_name) + " " + str(label_name) + '\n')

    print("processing done !")
    

if __name__ == "__main__":
    main()
