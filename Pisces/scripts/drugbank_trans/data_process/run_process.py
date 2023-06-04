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

        transductive_data_dir = f"/home/swang/xuhw/research-projects/data/drugbank_for_ddi/transductive/fold{f_i}/"
        save_dir = f"/home/swang/xuhw/research-projects/data/Pisces/drugbank_trans/fold{f_i}/"
        raw_train_dir = "{}{}".format(raw_data_dir, 'drugbank_id_smiles.csv')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_drug_dict = {}
        drugbank_csv = pd.read_csv(raw_train_dir)
        for idx, smiles in zip(tqdm(drugbank_csv['DrugID']), drugbank_csv['Smiles']):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
            all_drug_dict[idx] = smi_tokenizer(smiles)

        drug2id = {}
        id = 0
        for idx, smiles in zip(tqdm(drugbank_csv['DrugID']), drugbank_csv['Smiles']):
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
        
        train_csv_file = "{}{}".format(transductive_data_dir, 'train.csv')
        valid_csv_file = "{}{}".format(transductive_data_dir, 'valid.csv')
        test_csv_file = "{}{}".format(transductive_data_dir, 'test.csv')
        
        train_a_dir = "{}{}".format(save_dir, 'train.a')
        train_b_dir = "{}{}".format(save_dir, 'train.b')
        train_pair_dir = "{}{}".format(save_dir, 'train.pair')
        valid_a_dir = "{}{}".format(save_dir, 'valid.a')
        valid_b_dir = "{}{}".format(save_dir, 'valid.b')
        valid_pair_dir = "{}{}".format(save_dir, 'valid.pair')
        test_a_dir = "{}{}".format(save_dir, 'test.a')
        test_b_dir = "{}{}".format(save_dir, 'test.b')
        test_pair_dir = "{}{}".format(save_dir, 'test.pair')

        train_nega_dir = "{}{}".format(save_dir, 'train.nega')
        train_negb_dir = "{}{}".format(save_dir, 'train.negb')
        train_negpair_dir = "{}{}".format(save_dir, 'train.negpair')
        valid_nega_dir = "{}{}".format(save_dir, 'valid.nega')
        valid_negb_dir = "{}{}".format(save_dir, 'valid.negb')
        valid_negpair_dir = "{}{}".format(save_dir, 'valid.negpair')
        test_nega_dir = "{}{}".format(save_dir, 'test.nega')
        test_negb_dir = "{}{}".format(save_dir, 'test.negb')
        test_negpair_dir = "{}{}".format(save_dir, 'test.negpair')

        train_label_dir = "{}{}".format(save_dir, 'train.label')
        valid_label_dir = "{}{}".format(save_dir, 'valid.label')
        test_label_dir = "{}{}".format(save_dir, 'test.label')
        
        label_dict = set()
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
                tp_w.writelines(str(drug2id[a]) + '\t' + str(drug2id[b]) + '\n')

                tna_w.writelines(all_drug_dict[na] + '\n')
                tnb_w.writelines(all_drug_dict[nb] + '\n')
                tnp_w.writelines(str(drug2id[na]) + '\t' + str(drug2id[nb]) + '\n')

                # pdb.set_trace()
                tl.writelines(str(y) + '\n')
                label_dict.add(y)
        
            valid_csv = pd.read_csv(valid_csv_file)
            for a, b, na, nb, y in zip(tqdm(valid_csv['id1']), valid_csv['id2'], \
                valid_csv['neg_id1'], valid_csv['neg_id2'], valid_csv['y']):
                
                va_w.writelines(all_drug_dict[a] + '\n')
                vb_w.writelines(all_drug_dict[b] + '\n')
                vp_w.writelines(str(drug2id[a]) + '\t' + str(drug2id[b]) + '\n')

                vna_w.writelines(all_drug_dict[na] + '\n')
                vnb_w.writelines(all_drug_dict[nb] + '\n')
                vnp_w.writelines(str(drug2id[na]) + '\t' + str(drug2id[nb]) + '\n')

                vl.writelines(str(y) + '\n')
                label_dict.add(y)
        with open(test_a_dir, 'w') as tsa_w, open(test_nega_dir, 'w') as tsna_w, \
             open(test_b_dir, 'w') as tsb_w, open(test_negb_dir, 'w') as tsnb_w, \
             open(test_pair_dir, 'w') as tsp_w, open(test_negpair_dir, 'w') as tsnp_w, \
             open(test_label_dir, 'w') as tsl:
            test_csv = pd.read_csv(test_csv_file)
            for a, b, na, nb, y in zip(tqdm(test_csv['id1']), test_csv['id2'], \
                test_csv['neg_id1'], test_csv['neg_id2'], test_csv['y']):
                
                tsa_w.writelines(all_drug_dict[a] + '\n')
                tsb_w.writelines(all_drug_dict[b] + '\n')
                tsp_w.writelines(str(drug2id[a]) + '\t' + str(drug2id[b]) + '\n')

                tsna_w.writelines(all_drug_dict[na] + '\n')
                tsnb_w.writelines(all_drug_dict[nb] + '\n')
                tsnp_w.writelines(str(drug2id[na]) + '\t' + str(drug2id[nb]) + '\n')

                tsl.writelines(str(y) + '\n')
                label_dict.add(y)

        label_dict = list(label_dict)
        label_dict_dir  = "{}{}".format(save_dir, 'label.dict')
        label_dict = np.sort(np.asarray(label_dict))
        with open(label_dict_dir, 'w') as label_dict_w:
            for label_name in label_dict:
                label_dict_w.writelines(str(label_name) + " " + str(label_name) + '\n')

        print("processing done !")
    

if __name__ == "__main__":
    main()
