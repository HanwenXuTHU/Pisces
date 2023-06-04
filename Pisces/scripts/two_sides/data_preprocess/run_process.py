import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np


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
        print("processing start !")
        
        drug_names_dir = '/homes/gws/xuhw/research_projects/Pisces/data/two_sides/drug_names.csv'
        transductive_data_dir = "/data/xuhw/data/twosides_data/"
        tsd_bpe_data_dir = "/data/xuhw/data/Pisces/twosides/fold{}/".format(f_i)

        drug_id_dict = {}
        f = open(drug_names_dir, 'r')
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip().split('\t')
            drug_id_dict[int(line[0][3: ])] = str(idx)

        if not os.path.exists(tsd_bpe_data_dir):
            os.makedirs(tsd_bpe_data_dir)

        with open(os.path.join(tsd_bpe_data_dir, 'drug_name.dict'), 'w') as f1, \
            open(os.path.join(tsd_bpe_data_dir, 'drug_id.dict'), 'w') as f2:
            for key, value in drug_id_dict.items():
                f1.write(str(key) + ' ' + str(value) + '\n')
                f2.write(str(value) + ' ' + str(value) + '\n')
            f1.close()
            f2.close()

        '''smiles_file = "{}{}".format(transductive_data_dir, 'twosides.csv')

        all_drug_dict = {}
        drug_set = set()
        drug_pos_pair = {}
        drugbank_csv = pd.read_csv(smiles_file)
        for id_x1, x1, id_x2, x2 in zip(tqdm(drugbank_csv['ID1']), drugbank_csv['X1'], \
            drugbank_csv['ID2'], drugbank_csv['X2']):
            if id_x1 not in all_drug_dict.keys():
                x1_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(x1)))
                all_drug_dict[id_x1] = smi_tokenizer(x1_smiles)
            if id_x2 not in all_drug_dict.keys():
                x2_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(x2)))
                all_drug_dict[id_x2] = smi_tokenizer(x2_smiles)
            
            drug_set.add(id_x1)
            drug_set.add(id_x2)

            if id_x1 not in drug_pos_pair:
                drug_pos_pair[id_x1] = set([id_x2])
            else:
                drug_pos_pair[id_x1].add(id_x2)

            if id_x2 not in drug_pos_pair:
                drug_pos_pair[id_x2] = set([id_x1])
            else:
                drug_pos_pair[id_x2].add(id_x1)

        fold0_train = "{}{}".format(transductive_data_dir, 'pair_pos_neg_triples_train_fold{}.csv'.format(f_i))
        fold0_test = "{}{}".format(transductive_data_dir, 'pair_pos_neg_triples_test_fold{}.csv'.format(f_i))
        
        train_a_dir = "{}{}".format(tsd_bpe_data_dir, 'train.a')
        train_b_dir = "{}{}".format(tsd_bpe_data_dir, 'train.b')
        train_pair_dir = "{}{}".format(tsd_bpe_data_dir, 'train.pair')

        train_nega_dir = "{}{}".format(tsd_bpe_data_dir, 'train.nega')
        train_negb_dir = "{}{}".format(tsd_bpe_data_dir, 'train.negb')
        train_negpair_dir = "{}{}".format(tsd_bpe_data_dir, 'train.negpair')
        
        train_label_b_dir = "{}{}".format(tsd_bpe_data_dir, 'train.label')

        label_dict = {}
        unhit = 0
        with open(train_a_dir, 'w') as ta_w, open(train_nega_dir, 'w') as tna_w, \
            open(train_b_dir, 'w') as tb_w, open(train_negb_dir, 'w') as tnb_w, \
            open(train_pair_dir, 'w') as tp_w, open(train_negpair_dir, 'w') as tnp_w, \
            open(train_label_b_dir, 'w') as tl:
            
            f0t_csv = pd.read_csv(fold0_train)
            for a, b, y, n in zip(tqdm(f0t_csv['Drug1_ID']), f0t_csv['Drug2_ID'], f0t_csv['Y'], f0t_csv['Neg samples']):
                
                if np.random.rand() < 0.5:
                    nega = a
                    negb = n
                else:
                    negb = b
                    nega = n

                if a not in all_drug_dict or b not in all_drug_dict or \
                    nega not in all_drug_dict or negb not in all_drug_dict:
                    unhit += 1
                    continue

                ta_w.writelines(all_drug_dict[a] + '\n')
                tb_w.writelines(all_drug_dict[b] + '\n')
                tp_w.writelines(drug_id_dict[a] + '\t' + drug_id_dict[b] + '\n')

                tna_w.writelines(all_drug_dict[nega] + '\n')
                tnb_w.writelines(all_drug_dict[negb] + '\n')
                tnp_w.writelines(drug_id_dict[nega] + '\t' + drug_id_dict[negb] + '\n')

                tl.writelines(str(y) + '\n')
                if str(y) not in label_dict:
                    label_dict[str(y)] = len(label_dict)
        print(unhit)

        train_a_dir = "{}{}".format(tsd_bpe_data_dir, 'valid.a')
        train_b_dir = "{}{}".format(tsd_bpe_data_dir, 'valid.b')
        train_pair_dir = "{}{}".format(tsd_bpe_data_dir, 'valid.pair')

        train_nega_dir = "{}{}".format(tsd_bpe_data_dir, 'valid.nega')
        train_negb_dir = "{}{}".format(tsd_bpe_data_dir, 'valid.negb')
        train_negpair_dir = "{}{}".format(tsd_bpe_data_dir, 'valid.negpair')

        train_label_b_dir = "{}{}".format(tsd_bpe_data_dir, 'valid.label')

        unhit = 0
        with open(train_a_dir, 'w') as ta_w, open(train_nega_dir, 'w') as tna_w, \
            open(train_b_dir, 'w') as tb_w, open(train_negb_dir, 'w') as tnb_w, \
            open(train_pair_dir, 'w') as tp_w, open(train_negpair_dir, 'w') as tnp_w, \
            open(train_label_b_dir, 'w') as tl:
            
            f0t_csv = pd.read_csv(fold0_test)
            for a, b, y, n in zip(tqdm(f0t_csv['Drug1_ID']), f0t_csv['Drug2_ID'], f0t_csv['Y'], f0t_csv['Neg samples']):

                if np.random.rand() < 0.5:
                    nega = a
                    negb = n
                else:
                    negb = b
                    nega = n

                ta_w.writelines(all_drug_dict[a] + '\n')
                tb_w.writelines(all_drug_dict[b] + '\n')
                tp_w.writelines(drug_id_dict[a] + '\t' + drug_id_dict[b] + '\n')

                tna_w.writelines(all_drug_dict[nega] + '\n')
                tnb_w.writelines(all_drug_dict[negb] + '\n')
                tnp_w.writelines(drug_id_dict[nega] + '\t' + drug_id_dict[negb] + '\n')
                
                tl.writelines(str(y) + '\n')
                if str(y) not in label_dict:
                    label_dict[str(y)] = len(label_dict)
        print(unhit)
        
        label_dict_dir  = "{}{}".format(tsd_bpe_data_dir, 'label.dict')
        with open(label_dict_dir, 'w') as label_dict_w:
            for label_name, label_idx in label_dict.items():
                label_dict_w.writelines(label_name + " " + str(label_idx) + '\n')'''

        print("processing done !")
    
    
   
if __name__ == "__main__":
    main()
