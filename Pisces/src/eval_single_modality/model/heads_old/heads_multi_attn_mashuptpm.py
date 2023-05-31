from torch import layout, nn
from fairseq import utils
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pdb
from ..heads_ppi import DataPPI


class MultiModalitiesAttnMashupTpmHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 dv_input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout,
                 n_memory,
                 drug_dict: dict):

        super().__init__()
        
        self.cell_num = num_classes
        self.emb_dim = inner_dim
        self.n_hop = 2
        self.n_memory = n_memory
        self.n_heads = 8

        mashup_path = 'extension_data/cell_line_mashup.pkl'
        with open(mashup_path, 'rb') as f:
            self.cell_mashup = pickle.load(f)
        tpm_path = 'extension_data/cell_line_tpm.pkl'
        with open(tpm_path, 'rb') as f:
            self.cell_tpm = pickle.load(f)

        ppi_loader = DataPPI(
            aux_data_dir='baselines/GraphSynergy-master/data_ours_3fold',
            drug_target_path='baselines/GraphSynergy-master/data_ours_3fold/drug_protein.csv',
            n_hop=self.n_hop,
            n_memory=self.n_memory)

        self.cell_neighbor_set = ppi_loader.get_cell_neighbor_set()
        self.drug_neighbor_set = ppi_loader.get_drug_neighbor_set()
        node_num_dict = ppi_loader.get_node_num_dict()
        self.protein_num = node_num_dict['protein']

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * 2 * self.n_hop, self.emb_dim)

        self.transformer_proj_head = nn.Linear(input_dim, inner_dim)
        self.graph_proj_head = nn.Linear(dv_input_dim, inner_dim)
        self.mix_linear = nn.Linear(inner_dim * 2, inner_dim)
        self.layernorm_cell = nn.LayerNorm(inner_dim)
        self.layernorm_drug = nn.LayerNorm(inner_dim)

        self.mashup_encoder = nn.Linear(self._get_cell_dim(self.cell_mashup), inner_dim)
        self.tpm_encoder = nn.Linear(self._get_cell_dim(self.cell_tpm), inner_dim)

        self.used_modal = ['SMILES', 'Graph', '3D', 'Side effect', 'Drug Sensitivity (NCI60)', \
                          'Text', 'Drug Ontology', 'Drug target']
        self.raw_data_path = 'extension_data/drug_modalities.pkl'
        self.drug_dict = drug_dict
        self.id2drug = {v: k for k, v in self.drug_dict.items()}
        # load raw data
        with open(self.raw_data_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        self.is_modal = self.get_is_modal()
        for mod in self.used_modal:
            if 'Side effect' in mod:
                self.sider_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
            elif 'Drug Sensitivity (NCI60)' in mod:
                self.nci60_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
            elif 'Drug Ontology' in mod:
                self.dron_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
            elif 'Text' in mod:
                self.text_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
            elif '3D' in mod:
                self.drug_3d_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
        
        self.modal_agg = nn.MultiheadAttention(inner_dim, self.n_heads)
        self.cell_agg = nn.MultiheadAttention(inner_dim, self.n_heads)

        self.classifier_1 = nn.Sequential(
            nn.Linear(3 * inner_dim, inner_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim, 1)
        )

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def get_cell_emb(self, cell):
        np_cell = cell.squeeze().cpu().numpy().tolist()
        cell_mashup = torch.HalfTensor([self.cell_mashup[c].squeeze() for c in np_cell]).to(cell.device)
        cell_mashup = torch.functional.F.normalize(cell_mashup, dim=1)
        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cell.squeeze(1).cpu().numpy().tolist()]).to(cell.device))
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        cell_learnable = self._aggregation(cell_i_list)
        cell_learnable = self.layernorm_cell(cell_learnable)
        cell_mashup = self.mashup_encoder(cell_mashup)
        cell_mashup = self.layernorm_cell(cell_mashup)
        return cell_learnable, cell_mashup

    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, cells, pair=None, labels=None):

        cell_learnable, cell_mashup = self.get_cell_emb(cells)
        cell_all_modal = torch.cat([cell_learnable.unsqueeze(0), cell_mashup.unsqueeze(0)], dim=0)
        cell_embeddings = self.cell_agg(cell_all_modal, cell_all_modal, cell_all_modal)[0].mean(dim=0)
        # cell_all_modal [modal_num, batch_size, inner_dim]
        # randomly choose one modal for each sample from cell_all_modal
        cell_random_idx1 = torch.randint(0, cell_all_modal.shape[0], (cell_all_modal.shape[1],))
        cell_random_idx2 = torch.randint(0, cell_all_modal.shape[0], (cell_all_modal.shape[1],))
        cell_random_modal1 = cell_all_modal[cell_random_idx1, torch.arange(cell_all_modal.shape[1])]
        cell_random_modal2 = cell_all_modal[cell_random_idx2, torch.arange(cell_all_modal.shape[1])]

        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)

        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)

        # get the input of other modal
        other_modal_a, other_modal_b = self._get_pair_input(pair)
        out_a, out_b = self._get_pair_output(other_modal_a, other_modal_b)
        out_a['SMILES'], out_a['Graph'] = ta, ga
        out_b['SMILES'], out_b['Graph'] = tb, gb
        # heads other modal
        heads_all_modal = torch.cat([out_a[mod].unsqueeze(0) for mod in self.used_modal], dim=0)
        tail_all_modal = torch.cat([out_b[mod].unsqueeze(0) for mod in self.used_modal], dim=0)
        # get the attention mask according to the is_modal
        heads_mask, tail_mask = self.get_is_modal(pair)
        heads = self.modal_agg(heads_all_modal, heads_all_modal, heads_all_modal, key_padding_mask=heads_mask)[0]
        tails = self.modal_agg(tail_all_modal, tail_all_modal, tail_all_modal, key_padding_mask=tail_mask)[0]
        # get the mean of heads with heads_mask
        # heads: [n_modal, batch_size, inner_dim]
        # heads_mask: [batch_size, n_modal]
        heads_mask = (1 - heads_mask.half().permute(1, 0))
        tail_mask = (1 - tail_mask.half().permute(1, 0))
        heads = torch.sum(heads * heads_mask.unsqueeze(-1), dim=0) / torch.sum(heads_mask, dim=0).unsqueeze(-1)
        tails = torch.sum(tails * tail_mask.unsqueeze(-1), dim=0) / torch.sum(tail_mask, dim=0).unsqueeze(-1)
        # concat
        xc = torch.cat([heads, tails, cell_embeddings], dim=1)
        xc = self.classifier_1(xc)
        out = self.classifier_2(xc)

        '''combo_list = [[ta, tb], [ta, gb], [ga, tb], [ga, gb]]
        combo_idxes = np.random.choice(range(4), size=2, replace=False, p=[1/4] * 4)
        assert combo_idxes[0] != combo_idxes[1]'''
        random_a1, random_b1 = self._get_random_pairs(pair, out_a, out_b)
        random_a2, random_b2 = self._get_random_pairs(pair, out_a, out_b)

        xc_1_raw = torch.cat([random_a1, random_b1] + [cell_random_modal1], dim=1)
        xc_1 = self.classifier_1(xc_1_raw)
        sub_out_1 = self.classifier_2(xc_1)

        xc_2_raw = torch.cat([random_a2, random_b2] + [cell_random_modal2], dim=1)
        xc_2 = self.classifier_1(xc_2_raw)
        sub_out_2 = self.classifier_2(xc_2)

        cosine_ttgg = self.get_cosine_loss(xc_1_raw, xc_2_raw)
        consine = cosine_ttgg

        return out, consine, 0.5 * (sub_out_1 + sub_out_2)
    
    def get_cosine_loss(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = self.contrastive_loss(logits, targets)
        return loss

    def get_is_modal(self, pair=None):
        # judge if self have is_modal
        if not hasattr(self, 'is_modal'):
            is_modal = np.zeros([len(self.raw_data.keys()), len(self.used_modal)])
            for d in self.raw_data.keys():
                for mdl in self.used_modal:
                    if self.raw_data[d][mdl] is not None:
                        is_modal[self.drug_dict[d], self.used_modal.index(mdl)] = 1
            self.is_modal = is_modal
            return is_modal
        if pair is not None:
            np_pair = pair.cpu().numpy()
            pair_mask_a = 1 - self.is_modal[np_pair[:, 0], :]
            pair_mask_b = 1 - self.is_modal[np_pair[:, 1], :]
            pair_mask_a = torch.from_numpy(pair_mask_a).to(pair.device)
            pair_mask_b = torch.from_numpy(pair_mask_b).to(pair.device)
            return pair_mask_a.bool(), pair_mask_b.bool()
        else:
            return None

    def _get_modal_dim(self, mod):
        for d in self.raw_data.keys():
            if self.raw_data[d][mod] is not None:
                break
        return self.raw_data[d][mod].reshape(1, -1).shape[1]

    def _get_cell_dim(self, dict_data):
        for d in dict_data.keys():
            break
        return dict_data[d].reshape(1, -1).shape[1]

    def _get_neighbor_emb(self, neighbors):
        neighbors_emb_list = []
        for hop in range(self.n_hop):
            neighbors_emb_list.append(self.protein_embedding(neighbors[hop]))
        return neighbors_emb_list

    def _interaction_aggregation(self, neighbors_emb_list):
        interact_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim]
            neighbor_emb = neighbors_emb_list[hop]
            aggr_mean = torch.mean(neighbor_emb, dim=1)
            aggr_max = torch.max(neighbor_emb, dim=1).values
            interact_list.append(torch.cat([aggr_mean, aggr_max], dim=-1))
        
        return interact_list

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings

    def _get_pair_input(self, pair):
        '''
        get the input vectors for each modal
        pair: [batch_size, 2]
        '''
        np_pairs = pair.cpu().numpy()
        other_modal_a, other_modal_b = {}, {}
        for mod in self.used_modal:
            if mod in ['Drug target']:
                drug_a_neighbors, drug_b_neighbors = [], []
                for hop in range(self.n_hop):
                    drug_a_neighbors.append(torch.LongTensor([self.drug_neighbor_set[self.id2drug[p]][hop] \
                                                            for p in pair.cpu().numpy()[:, 0]]).to(pair.device))
                    drug_b_neighbors.append(torch.LongTensor([self.drug_neighbor_set[self.id2drug[p]][hop] \
                                                            for p in pair.cpu().numpy()[:, 1]]).to(pair.device))
                other_modal_a[mod] = drug_a_neighbors
                other_modal_b[mod] = drug_b_neighbors
                continue
            if mod in ['SMILES', 'Graph']:
                continue
            impute_vector = -1000*np.ones(self._get_modal_dim(mod))
            f_a, f_b = [], []
            for i in np_pairs[:, 0]:
                d_a_name = self.id2drug[i]
                if self.raw_data[d_a_name][mod] is not None:
                    f_a.append(self.raw_data[d_a_name][mod].reshape(-1))
                else:
                    f_a.append(impute_vector)
            for i in np_pairs[:, 1]:
                d_b_name = self.id2drug[i]
                if self.raw_data[d_b_name][mod] is not None:
                    f_b.append(self.raw_data[d_b_name][mod].reshape(-1))
                else:
                    f_b.append(impute_vector)
            
            f_a, f_b = np.array(f_a, dtype=float), np.array(f_b, dtype=float)
            #impute the data, set the missing vectors to the mean of the existing vectors
            f_a, f_b = self._impute_input_data(f_a), self._impute_input_data(f_b)
            f_a = torch.from_numpy(f_a).half().to(pair.device)
            f_b = torch.from_numpy(f_b).half().to(pair.device)
            other_modal_a[mod], other_modal_b[mod] = f_a, f_b
        return other_modal_a, other_modal_b

    def _get_pair_output(self, other_modal_a, other_modal_b):
        out_a, out_b = {}, {}
        for mod in self.used_modal:
            if mod in ['SMILES', 'Graph']:
                continue
            if 'Side effect' in mod:
                out_a[mod] = self.sider_encoder(other_modal_a[mod])
                out_b[mod] = self.sider_encoder(other_modal_b[mod])
            elif 'Drug Sensitivity (NCI60)' in mod:
                out_a[mod] = self.nci60_encoder(other_modal_a[mod])
                out_b[mod] = self.nci60_encoder(other_modal_b[mod]) 
            elif 'Drug Ontology' in mod:
                out_a[mod] = self.dron_encoder(other_modal_a[mod])
                out_b[mod] = self.dron_encoder(other_modal_b[mod])
            elif 'Text' in mod:
                out_a[mod] = self.text_encoder(other_modal_a[mod])
                out_b[mod] = self.text_encoder(other_modal_b[mod])
            elif '3D' in mod:
                out_a[mod] = self.drug_3d_encoder(other_modal_a[mod])
                out_b[mod] = self.drug_3d_encoder(other_modal_b[mod])
            elif 'Drug target' in mod:
                target_a_neighbors_emb_list = self._get_neighbor_emb(other_modal_a[mod])
                a_list = self._interaction_aggregation(target_a_neighbors_emb_list)
                target_a_embeddings = self._aggregation(a_list)
                target_a_embeddings = self.layernorm_drug(target_a_embeddings)
                target_b_neighbors_emb_list = self._get_neighbor_emb(other_modal_b[mod])
                b_list = self._interaction_aggregation(target_b_neighbors_emb_list)
                target_b_embeddings = self._aggregation(b_list)
                target_b_embeddings = self.layernorm_drug(target_b_embeddings)
                out_a[mod], out_b[mod] = target_a_embeddings, target_b_embeddings
        return out_a, out_b

    def _impute_input_data(self, f_array):
        for i in range(f_array.shape[1]):
            col = f_array[:, i]
            if np.sum(col == -1000) == 0:
                continue
            else:
                col_mean = np.mean(col[col != -1000])
                if np.isnan(col_mean):
                    col_mean = -1000
                col[col == -1000] = col_mean
            f_array[:, i] = col
        return f_array

    def _get_random_pairs(self, pairs, out_a, out_b):
        random_a, random_b = [], []
        np_pairs = pairs.cpu().numpy()
        for i in range(len(np_pairs)):
            a_id, b_id = np_pairs[i, 0], np_pairs[i, 1]
            a_mod_list, b_mod_list = self.is_modal[a_id, :], self.is_modal[b_id, :]
            a_mod_list = np.asarray(self.used_modal)[a_mod_list == 1]
            b_mod_list = np.asarray(self.used_modal)[b_mod_list == 1]
            # randomly choose a modal from the modal list
            a_mod, b_mod = np.random.choice(a_mod_list), np.random.choice(b_mod_list)
            a_feat, b_feat = out_a[a_mod][i, :], out_b[b_mod][i, :]
            random_a.append(a_feat.reshape(1, -1))
            random_b.append(b_feat.reshape(1, -1))
        random_a, random_b = torch.cat(random_a, 0), torch.cat(random_b, 0)
        return random_a, random_b
        
