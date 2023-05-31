from torch import layout, nn
from fairseq import utils
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pdb
from .heads_ppi import DataPPI


class MultiHeadTestTopK(nn.Module):
    
    def __init__(self,
                 input_dim,
                 dv_input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout,
                 n_memory,
                 drug_dict: dict,
                 seen_drugs: set,
                 topk=2,
                 raw_data_path='data/two_sides/drug_modalities.pkl',
                 drug_target_path='data/two_sides/drug_target.csv',
                 fp16=True):

        super().__init__()
        
        self.num_classes = num_classes
        self.inner_dim = inner_dim
        self.emb_dim = inner_dim
        self.n_hop = 2
        self.n_memory = n_memory
        self.topk = topk
        self.fp16 = fp16
        self.second_order = True

        ppi_loader = DataPPI(
            aux_data_dir='data/ppi',
            drug_target_path=drug_target_path,
            n_hop=self.n_hop,
            n_memory=self.n_memory)

        self.seen_drugs = seen_drugs
        self.cell_neighbor_set = ppi_loader.get_cell_neighbor_set()
        self.drug_neighbor_set = ppi_loader.get_drug_neighbor_set()
        node_num_dict = ppi_loader.get_node_num_dict()
        self.protein_num = node_num_dict['protein']
        # add neighbor set for drug without any target protein
        for d in drug_dict.keys():
            if d not in self.drug_neighbor_set:
                self.drug_neighbor_set[d] = []
                for h in range(self.n_hop):
                    self.drug_neighbor_set[d].append([])
                    for m in range(self.n_memory):
                        self.drug_neighbor_set[d][h].append(self.protein_num + drug_dict[d])

        self.protein_embedding = nn.Embedding(self.protein_num + len(drug_dict.keys()), self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * 2 * self.n_hop, self.emb_dim)

        self.transformer_proj_head = nn.Linear(input_dim, inner_dim)
        self.graph_proj_head = nn.Linear(dv_input_dim, inner_dim)
        #self.mix_linear = nn.MultiheadAttention(2 * inner_dim, 4, dropout=pooler_dropout, batch_first=True)
        #self.mix_linear_head = nn.Sequential(
        #    nn.LayerNorm(2 * inner_dim),
        #    nn.ReLU(),
        #)
        self.layernorm_drug = nn.LayerNorm(inner_dim)
        self.layernorm_cell = nn.LayerNorm(inner_dim)

        self.used_modal = ['SMILES', 'Graph', '3D', 'Side effect', 'Drug Sensitivity (NCI60)', \
                          'Text', 'Drug Ontology', 'Drug target']
        self.raw_data_path = raw_data_path
        self.drug_dict = drug_dict
        self.id2drug = {v: k for k, v in self.drug_dict.items()}
        self.drug_n = len(self.drug_dict)
        # load raw data
        with open(self.raw_data_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        self.is_modal = self.get_is_modal()
        for mod in self.used_modal:
            if 'Side effect' in mod:
                self.sider_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
                self.sider_unk_emb = nn.Embedding(self.drug_n, self._get_modal_dim(mod))
            elif 'Drug Sensitivity (NCI60)' in mod:
                self.nci60_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
                self.nci60_unk_emb = nn.Embedding(self.drug_n, self._get_modal_dim(mod))
            elif 'Drug Ontology' in mod:
                self.dron_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
                self.dron_unk_emb = nn.Embedding(self.drug_n, self._get_modal_dim(mod))
            elif 'Text' in mod:
                self.text_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)
            elif '3D' in mod:
                self.drug_3d_encoder = nn.Linear(self._get_modal_dim(mod), inner_dim)

        self.classifier_1 = nn.Sequential(
            nn.Linear(3 * inner_dim, 2 * inner_dim),
            nn.Tanh(), 
            nn.Dropout(p=pooler_dropout),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim, 1)
        )

        if self.second_order:
            self.second_order_mix = nn.Sequential(
                nn.Linear(4 * inner_dim, 2 * inner_dim),
                nn.Tanh(),
                nn.Dropout(p=pooler_dropout),
            )

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def get_pred_batch(self, out_a_dict, out_b_dict, cell_embeddings):
        all_combo_emb = []
        for mod1 in self.used_modal:
            for mod2 in self.used_modal:
                all_combo_emb.append(self.classifier_1(torch.cat([out_a_dict[mod1], out_b_dict[mod2], cell_embeddings], dim=1)))
        all_pred = [a.unsqueeze(1) for a in all_combo_emb]
        all_pred = torch.cat(all_pred, dim=1)
        all_pred = self.classifier_2(all_pred).squeeze(-1)
        # select top K prediction and average
        top_k_pred, _ = torch.topk(all_pred, k=self.topk, dim=1)
        out = torch.mean(top_k_pred, dim=1, keepdim=True)
        return out, all_combo_emb
    
    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, cells, pair=None, labels=None):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(drug_a.device)) 
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        cell_embeddings = self._aggregation(cell_i_list)
        cell_embeddings = self.layernorm_cell(cell_embeddings)

        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)

        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)

        # get the input of other modal
        other_modal_a, other_modal_b = self._get_pair_input(pair)
        out_a_dict, out_b_dict = self._get_pair_output(other_modal_a, other_modal_b)
        out_a_dict['SMILES'], out_a_dict['Graph'] = ta, ga
        out_b_dict['SMILES'], out_b_dict['Graph'] = tb, gb
        out, all_combo_emb = self.get_pred_batch(out_a_dict, out_b_dict, cell_embeddings)

        if self.second_order:
            combo_idxes = np.random.choice(range(len(all_combo_emb)), size=4, replace=False, \
                                       p=[1/len(all_combo_emb)] * len(all_combo_emb))
            xc_s1 = torch.cat([all_combo_emb[combo_idxes[0]], all_combo_emb[combo_idxes[1]]], dim=1)
            xc_s2 = torch.cat([all_combo_emb[combo_idxes[2]], all_combo_emb[combo_idxes[3]]], dim=1)
            
            xc_1 = self.second_order_mix(xc_s1)
            xc_2 = self.second_order_mix(xc_s2)
            consine = self.get_cosine_loss(xc_1, xc_2)
            sub_out_1 = self.classifier_2(xc_1)
            sub_out_2 = self.classifier_2(xc_2)
        else:
            combo_idxes = np.random.choice(range(len(all_combo_emb)), size=2, replace=False, \
                                        p=[1/len(all_combo_emb)] * len(all_combo_emb))
            assert combo_idxes[0] != combo_idxes[1]
            sub_out_1 = self.classifier_2(all_combo_emb[combo_idxes[0]])
            sub_out_2 = self.classifier_2(all_combo_emb[combo_idxes[1]])
            xc_1 = all_combo_emb[combo_idxes[0]].squeeze()
            xc_2 = all_combo_emb[combo_idxes[1]].squeeze()

        consine = self.get_cosine_loss(xc_1, xc_2)

        return out.reshape(-1, 1), consine, 0.5 * (sub_out_1 + sub_out_2).reshape(-1, 1)
    
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
            pair_mask_a = 1 - self.is_modal[np_pair[:, 0], 1:]
            pair_mask_b = 1 - self.is_modal[np_pair[:, 1], 1:]
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
            f_a, f_b = [], []
            for i in np_pairs[:, 0]:
                d_a_name = self.id2drug[i]
                if self.raw_data[d_a_name][mod] is not None:
                    a_i = self.raw_data[d_a_name][mod].reshape([1, -1]).astype(np.float16)
                    a_i = torch.from_numpy(a_i).to(pair.device)
                    if self.fp16: a_i = a_i.half()
                    f_a.append(a_i)
                else:
                    emb_idx = torch.LongTensor([i]).to(pair.device)
                    a_i = self._get_unk_emb(mod, emb_idx)
                    if self.fp16: a_i = a_i.half()
                    f_a.append(a_i)
            for i in np_pairs[:, 1]:
                d_b_name = self.id2drug[i]
                if self.raw_data[d_b_name][mod] is not None:
                    b_i = self.raw_data[d_b_name][mod].reshape([1, -1]).astype(np.float16)
                    b_i = torch.from_numpy(b_i).to(pair.device)
                    if self.fp16: b_i = b_i.half()
                    f_b.append(b_i)
                else:
                    emb_idx = torch.LongTensor([i]).to(pair.device)
                    b_i = self._get_unk_emb(mod, emb_idx)
                    if self.fp16: b_i = b_i.half()
                    f_b.append(b_i)
            
            f_a = torch.cat(f_a, dim=0)
            f_b = torch.cat(f_b, dim=0)
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

    def _get_unk_emb(self, mod, emb_idx):
        if 'Side effect' in mod:
            return self.sider_unk_emb(emb_idx)
        elif 'Drug Sensitivity (NCI60)' in mod:
            return self.nci60_unk_emb(emb_idx)
        elif 'Drug Ontology' in mod:
            return self.dron_unk_emb(emb_idx)
        else:
            return None 
        
