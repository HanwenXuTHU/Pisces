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


class MultiHeadTree(nn.Module):
    
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
                 tree_level=8,
                 fp16=False):

        super().__init__()
        
        self.num_classes = num_classes
        self.inner_dim = inner_dim
        self.emb_dim = inner_dim
        self.n_hop = 2
        self.n_memory = n_memory
        self.topk = topk
        self.fp16 = fp16

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
        self.rel_emb = nn.Embedding(self.num_classes, inner_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * 2 * self.n_hop, self.emb_dim)
        self.rels_linear = nn.Sequential(
            nn.Linear(2*self.emb_dim, inner_dim),
            nn.Tanh(),
            nn.Dropout(p=pooler_dropout),
        )

        self.transformer_proj_head = nn.Linear(input_dim, inner_dim)
        self.graph_proj_head = nn.Linear(dv_input_dim, inner_dim)
        self.mix_linear = nn.Linear(inner_dim * 2, inner_dim)
        self.layernorm_drug = nn.LayerNorm(inner_dim)
        self.layernorm_cell = nn.LayerNorm(inner_dim)

        self.used_modal = ['SMILES', 'Graph', '3D', 'Side effect', 'Drug Sensitivity (NCI60)', \
                          'Text', 'Drug Ontology', 'Drug target']
        self.num_tree_level = tree_level
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
        
        self.modal_agg = nn.MultiheadAttention(inner_dim, 4)
        self.modal_unk_emb = nn.Embedding(len(self.used_modal)*len(self.raw_data.keys()), inner_dim)

        self.classifier_1_list = nn.ModuleList()
        for i in range(len(self.used_modal)):
            self.classifier_1_list.append(nn.Sequential(
                nn.Linear(2 * inner_dim * (i + 1), 2 * inner_dim),
                nn.Tanh(), 
                nn.Dropout(p=pooler_dropout),
            ))
        
        self.classifier_2 = nn.Sequential(
            nn.Linear(inner_dim * 3, 2 * inner_dim),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim, 1)
        )

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def get_mod_mask(self, a_rank, b_rank, pair):
        np_pair = pair.cpu().numpy()
        np_a_rank, np_b_rank = a_rank.cpu().numpy(), b_rank.cpu().numpy()
        is_modal_a = self.is_modal[np_pair[:, 0, np.newaxis], np_a_rank]
        is_modal_b = self.is_modal[np_pair[:, 1, np.newaxis], np_b_rank]
        is_seen_a = np.isin(np_pair[:, 0], list(self.seen_drugs)).astype(np.int32)
        is_seen_a = np.repeat(is_seen_a[:, np.newaxis], np_a_rank.shape[1], axis=1)
        is_seen_b = np.isin(np_pair[:, 1], list(self.seen_drugs)).astype(np.int32)
        is_seen_b = np.repeat(is_seen_b[:, np.newaxis], np_b_rank.shape[1], axis=1)
        mask_a = (is_modal_a + is_seen_a) >= 1
        mask_b = (is_modal_b + is_seen_b) >= 1
        return torch.from_numpy(mask_a).to(pair.device), torch.from_numpy(mask_b).to(pair.device)
    
    def get_pred_batch(self, a, b, cells, tree_level=0):
        if len(a.shape) == 2:
            a, b = a.unsqueeze(1), b.unsqueeze(1)
        x = self.classifier_1_list[tree_level](torch.cat([a, b], dim=-1))
        return self.x_cells_pred(x, cells)
    
    def x_cells_pred(self, x, cells):
        cells_batch = cells.repeat(1, x.shape[1], 1)
        input = torch.cat([x, cells_batch], dim=-1)
        # dot product along the last dimension
        pred = self.classifier_2(input)
        return pred.squeeze(), x.squeeze()

    def get_mod_rank(self, out_a, out_b, cells):
        a_mod_scores = torch.zeros((cells.shape[0], len(self.used_modal))).to(cells.device)
        b_mod_scores = torch.zeros((cells.shape[0], len(self.used_modal))).to(cells.device)
        if self.fp16:
            a_mod_scores = a_mod_scores.half()
            b_mod_scores = b_mod_scores.half()
        for mod1 in self.used_modal:
            for mod2 in self.used_modal:
                pred, _ = self.get_pred_batch(out_a[mod1], out_b[mod2], cells)
                a_mod_scores[:, self.used_modal.index(mod1)] += pred.squeeze()
                b_mod_scores[:, self.used_modal.index(mod2)] += pred.squeeze()
        a_rank = torch.argsort(a_mod_scores, dim=-1, descending=True)
        b_rank = torch.argsort(b_mod_scores, dim=-1, descending=True)
        return a_rank, b_rank
    
    def rank_the_mod(self, out_a_dict, out_b_dict, a_rank, b_rank):
        out_a = torch.stack([out_a_dict[mod] for mod in self.used_modal], dim=1)
        out_a = torch.gather(out_a, 1, a_rank.unsqueeze(-1).repeat(1, 1, out_a.shape[-1])).squeeze(1)
        out_b = torch.stack([out_b_dict[mod] for mod in self.used_modal], dim=1)
        out_b = torch.gather(out_b, 1, b_rank.unsqueeze(-1).repeat(1, 1, out_b.shape[-1])).squeeze(1)
        return out_a, out_b
    
    def get_upper_level_input(self, input, mask=None):
        output = torch.cat([input[:, :-1, :], input[:, 1:, -self.inner_dim: ]], dim=2)
        if mask is not None:
            mask_output = torch.multiply(mask[:, :-1], mask[:, 1:])
            return output, mask_output
        return output

    def get_expand_indices(self, batch_size, num_mod, inner_dim):
        ind_a_l1 = torch.repeat_interleave(torch.arange(num_mod), num_mod).reshape([1, -1, 1]).repeat(batch_size, 1, inner_dim)
        ind_b_l1 = torch.arange(num_mod).repeat(num_mod, 1).reshape(-1).reshape([1, -1, 1]).repeat(batch_size, 1, inner_dim)
        return ind_a_l1, ind_b_l1
    
    def expand_ab(self, a, b, a_mask=None, b_mask=None):
        ind_a, ind_b = self.get_expand_indices(a.shape[0], a.shape[1], a.shape[-1])
        ind_a, ind_b = ind_a.to(a.device), ind_b.to(a.device)
        a_exp, b_exp = torch.gather(a, 1, ind_a), torch.gather(b, 1, ind_b)
        if a_mask is not None:
            a_exp_mask = torch.gather(a_mask, 1, ind_a[:, :, 0])
            b_exp_mask = torch.gather(b_mask, 1, ind_b[:, :, 0])
            return a_exp, b_exp, a_exp_mask, b_exp_mask
        return a_exp, b_exp

    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, cells, pair=None, labels=None):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(drug_a.device)) 
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        cell_embeddings = self._aggregation(cell_i_list)
        cell_embeddings = self.layernorm_cell(cell_embeddings)
        cell_embeddings = cell_embeddings.unsqueeze(1)

        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)

        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)

        # get the input of other modal
        other_modal_a, other_modal_b = self._get_pair_input(pair)
        out_a_dict, out_b_dict = self._get_pair_output(other_modal_a, other_modal_b)
        out_a_dict['SMILES'], out_a_dict['Graph'] = ta, ga
        out_b_dict['SMILES'], out_b_dict['Graph'] = tb, gb
        all_combo_emb, all_pred = [], []
        all_mask = []
        a_rank, b_rank = self.get_mod_rank(out_a_dict, out_b_dict, cell_embeddings)
        a_mask, b_mask = self.get_mod_mask(a_rank, b_rank, pair)
        a_i, b_i = self.rank_the_mod(out_a_dict, out_b_dict, a_rank, b_rank)
        # perform the tree like generation
        # generate the first level indices
        for i in range(self.num_tree_level):
            a_exp_i, b_exp_i, a_exp_mask, b_exp_mask = self.expand_ab(a_i, b_i, a_mask, b_mask)
            pred, x_emb = self.get_pred_batch(a_exp_i, b_exp_i, cell_embeddings, tree_level=i)
            if len(pred.shape) == 1: pred, x_emb = pred.unsqueeze(-1), x_emb.unsqueeze(1)
            pair_mask = a_exp_mask * b_exp_mask
            # the second level of the tree
            a_i, a_mask = self.get_upper_level_input(a_i, a_mask)
            b_i, b_mask = self.get_upper_level_input(b_i, b_mask)
            all_combo_emb.append(x_emb)
            #
            all_pred.append(pred)
            all_mask.append(pair_mask)
            # rank pair_mask based on top_idx
        all_combo_emb = torch.cat(all_combo_emb, dim=1)
        all_pred = torch.cat(all_pred, dim=1)
        all_mask = torch.cat(all_mask, dim=1)

        out, top_idx = torch.topk(all_pred, k=len(self.used_modal), dim=1)
        top_mask = torch.gather(all_mask, 1, top_idx)
        out = torch.mean(out, dim=1, keepdim=True)

        combo_idxes = np.random.choice(range(all_combo_emb.size(1)), size=2, replace=False, \
                                       p=[1/all_combo_emb.size(1)] * all_combo_emb.size(1))
        assert combo_idxes[0] != combo_idxes[1]
        sub_out_1, xc_1 = self.x_cells_pred(all_combo_emb[:, combo_idxes[0], :].unsqueeze(1), cell_embeddings)
        sub_out_2, xc_2 = self.x_cells_pred(all_combo_emb[:, combo_idxes[1], :].unsqueeze(1), cell_embeddings)

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
                    a_i = self.raw_data[d_a_name][mod].reshape([1, -1]).astype(np.float32)
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
                    b_i = self.raw_data[d_b_name][mod].reshape([1, -1]).astype(np.float32)
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
        encoders = {
            'Side effect': self.sider_encoder,
            'Drug Sensitivity (NCI60)': self.nci60_encoder,
            'Drug Ontology': self.dron_encoder,
            'Text': self.text_encoder,
            '3D': self.drug_3d_encoder
        }
        
        out_a = {}
        out_b = {}
        
        for mod in self.used_modal:
            if mod in ['SMILES', 'Graph']:
                continue
            elif 'Drug target' in mod:
                target_a_neighbors_emb_list = self._get_neighbor_emb(other_modal_a[mod])
                a_list = self._interaction_aggregation(target_a_neighbors_emb_list)
                target_a_embeddings = self._aggregation(a_list)
                target_a_embeddings = self.layernorm_drug(target_a_embeddings)
                target_b_neighbors_emb_list = self._get_neighbor_emb(other_modal_b[mod])
                b_list = self._interaction_aggregation(target_b_neighbors_emb_list)
                target_b_embeddings = self._aggregation(b_list)
                target_b_embeddings = self.layernorm_drug(target_b_embeddings)
                out_a[mod] = target_a_embeddings
                out_b[mod] = target_b_embeddings
            else:
                out_a[mod] = encoders[mod](other_modal_a[mod])
                out_b[mod] = encoders[mod](other_modal_b[mod])
                
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
        
