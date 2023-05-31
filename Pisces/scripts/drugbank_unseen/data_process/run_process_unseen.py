import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np
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
        print("processing start !")

        raw_data_dir = "/home/swang/xuhw/research-projects/data/drugbank_for_ddi/transductive/"

        ind_unseen_data_dir = f"/home/swang/xuhw/research-projects/data/drugbank_for_ddi/ind_unseen/fold{f_i}/"
        save_dir = f"/home/swang/xuhw/research-projects/data/Pisces/drugbank_unseen/fold{f_i}/"
        raw_train_dir = "{}{}".format(raw_data_dir, 'drugbank_id_smiles.csv')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_drug_dict = {}
        drugbank_csv = pd.read_csv(raw_train_dir)
        for idx, smiles in zip(tqdm(drugbank_csv['DrugID']), drugbank_csv['Smiles']):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
            all_drug_dict[idx] = smi_tokenizer(smiles)

        drug_id_dict = {}
        for idx, drug_name in enumerate(all_drug_dict):
            drug_id_dict[drug_name] = str(idx)
        
        with open(os.path.join(save_dir, 'drug_name.dict'), 'w') as f1, \
            open(os.path.join(save_dir, 'drug_id.dict'), 'w') as f2:
            for key, value in drug_id_dict.items():
                f1.write(str(key) + ' ' + str(value) + '\n')
                f2.write(str(value) + ' ' + str(value) + '\n')
            f1.close()
            f2.close()
        
        train_csv_file = "{}{}".format(ind_unseen_data_dir, 'train.csv')
        valid_csv_file = "{}{}".format(ind_unseen_data_dir, 'valid.csv')
        test_csv_file = "{}{}".format(ind_unseen_data_dir, 'test.csv')
        
        train_a_dir = "{}{}".format(save_dir, 'train.a')
        train_b_dir = "{}{}".format(save_dir, 'train.b')
        train_pair_dir = "{}{}".format(save_dir, 'train.pair')
        valid_a_dir = "{}{}".format(save_dir, 'valid.a')
        valid_b_dir = "{}{}".format(save_dir, 'valid.b')
        valid_pair_dir = "{}{}".format(save_dir, 'valid.pair')

        train_nega_dir = "{}{}".format(save_dir, 'train.nega')
        train_negb_dir = "{}{}".format(save_dir, 'train.negb')
        train_negpair_dir = "{}{}".format(save_dir, 'train.negpair')
        valid_nega_dir = "{}{}".format(save_dir, 'valid.nega')
        valid_negb_dir = "{}{}".format(save_dir, 'valid.negb')
        valid_negpair_dir = "{}{}".format(save_dir, 'valid.negpair')

        train_label_dir = "{}{}".format(save_dir, 'train.label')
        valid_label_dir = "{}{}".format(save_dir, 'valid.label')
        
        label_dict = {}
        with open(train_a_dir, 'w') as ta_w, open(train_nega_dir, 'w') as tna_w, \
            open(train_b_dir, 'w') as tb_w, open(train_negb_dir, 'w') as tnb_w, \
            open(train_pair_dir, 'w') as tp_w, open(train_negpair_dir, 'w') as tnp_w, \
            open(valid_a_dir, 'w') as va_w, open(valid_nega_dir, 'w') as vna_w, \
            open(valid_b_dir, 'w') as vb_w, open(valid_negb_dir, 'w') as vnb_w, \
            open(valid_pair_dir, 'w') as vp_w, open(valid_negpair_dir, 'w') as vnp_w, \
            open(train_label_dir, 'w') as tl, open(valid_label_dir, 'w') as vl:
            
            train_csv = pd.read_csv(train_csv_file)
            for a, b, na, nb, y in zip(tqdm(train_csv['id1']), train_csv['id2'], \
                train_csv['neg_id1'], train_csv['neg_id2'], train_csv['y']):
                
                ta_w.writelines(all_drug_dict[a] + '\n')
                tb_w.writelines(all_drug_dict[b] + '\n')
                tp_w.writelines(drug_id_dict[a] + '\t' + drug_id_dict[b] + '\n')

                tna_w.writelines(all_drug_dict[na] + '\n')
                tnb_w.writelines(all_drug_dict[nb] + '\n')
                tnp_w.writelines(drug_id_dict[na] + '\t' + drug_id_dict[nb] + '\n')

                # pdb.set_trace()
                tl.writelines(str(y) + '\n')
                if str(y) not in label_dict:
                    label_dict[str(y)] = len(label_dict)
        
            valid_csv = pd.read_csv(valid_csv_file)
            for a, b, na, nb, y in zip(tqdm(valid_csv['id1']), valid_csv['id2'], \
                valid_csv['neg_id1'], valid_csv['neg_id2'], valid_csv['y']):
                
                va_w.writelines(all_drug_dict[a] + '\n')
                vb_w.writelines(all_drug_dict[b] + '\n')
                vp_w.writelines(drug_id_dict[a] + '\t' + drug_id_dict[b] + '\n')

                vna_w.writelines(all_drug_dict[na] + '\n')
                vnb_w.writelines(all_drug_dict[nb] + '\n')
                vnp_w.writelines(drug_id_dict[na] + '\t' + drug_id_dict[nb] + '\n')

                vl.writelines(str(y) + '\n')
                if str(y) not in label_dict:
                    label_dict[str(y)] = len(label_dict)

        label_dict_dir  = "{}{}".format(save_dir, 'label.dict')
        with open(label_dict_dir, 'w') as label_dict_w:
            for label_name, label_idx in label_dict.items():
                label_dict_w.writelines(label_name + " " + str(label_idx) + '\n')

        print("processing done !")
    

if __name__ == "__main__":
    main()
