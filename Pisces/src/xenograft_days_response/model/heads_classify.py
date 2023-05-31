from torch import layout, nn
from fairseq import utils
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pdb
from torch.autograd import Variable
from .heads_ppi import XenograftDataPPI
from .heads_wta import Heads_WTA


def time_emb(self, time_dim, max_len=1000):
    
    # Compute the positional encodings once in log space.
    pe = np.zeros([max_len, time_dim])
    position = np.arange(0, max_len).reshape(max_len, 1)
    div_term = np.exp(np.arange(0, time_dim, 2) * -(math.log(10000.0) / time_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
        

def get_neighbor_set(n_memory, drug_target_path, drug_dict):
    n_hop=2
    ppi_loader = XenograftDataPPI(
            aux_data_dir='data/ppi',
            drug_target_path=drug_target_path,
            n_hop=n_hop,
            n_memory=n_memory)

    gene_exp_neighbor_set = ppi_loader.get_c_neighbor_set()
    drug_neighbor_set = ppi_loader.get_drug_neighbor_set()
    node_num_dict = ppi_loader.get_node_num_dict()
    protein_num = node_num_dict['protein']
    # add neighbor set for drug without any target protein
    for d in drug_dict.keys():
        if d not in drug_neighbor_set:
            drug_neighbor_set[d] = []
            for h in range(n_hop):
                drug_neighbor_set[d].append([])
                for m in range(n_memory):
                    drug_neighbor_set[d][h].append(protein_num + drug_dict[d])
    return gene_exp_neighbor_set, drug_neighbor_set, protein_num


class HeadsClassify(nn.Module):
    
    def __init__(self,
                 input_dim,
                 dv_input_dim,
                 inner_dim,
                 num_classes,
                 pooler_dropout,
                 n_memory,
                 drug_dict: dict,
                 topk=2,
                 raw_data_path='',
                 drug_target_path='',
                 fp16=True,
                 wta_linear=True):

        super().__init__()
        
        self.num_classes = num_classes
        self.inner_dim = inner_dim
        self.emb_dim = inner_dim
        self.n_hop = 2
        self.time_dim = 200
        self.n_memory = n_memory
        self.topk = topk
        self.fp16 = fp16
        self.out_modal = ['SMILES', 'Graph', '3D', 'Side effect', 'Drug Sensitivity (NCI60)', \
                          'Text', 'Drug Ontology', 'Drug target']

        # load data
        self.raw_data_path = raw_data_path
        with open(self.raw_data_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        self.drug_dict = drug_dict
        self.id2drug = {v: k for k, v in self.drug_dict.items()}
        self.drug_n = len(self.drug_dict)
        self.is_modal = self.get_is_modal()
        self.time_emb = time_emb(self, time_dim=self.time_dim, max_len=1000)

        self.gene_exp_neighbor_set, self.drug_neighbor_set, self.protein_num = get_neighbor_set(n_memory, drug_target_path, drug_dict)

        self.set_nn(input_dim, dv_input_dim, inner_dim, pooler_dropout, wta_linear, self.drug_n)

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def set_nn(self, input_dim, dv_input_dim, inner_dim, pooler_dropout, wta_linear, drug_n):
        self.protein_embedding = nn.Embedding(self.protein_num + drug_n, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * 2 * self.n_hop, self.inner_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(pooler_dropout)
        self.transformer_proj_head = nn.Linear(input_dim, inner_dim)
        self.graph_proj_head = nn.Linear(dv_input_dim, inner_dim)
        self.layernorm_drug = nn.LayerNorm(inner_dim)
        self.layernorm_gene_exp = nn.LayerNorm(inner_dim)
        self.wta_layer = Heads_WTA(len(self.out_modal)**2, topk=self.topk, is_linear=wta_linear, is_mask=False)
        self.time_layer = nn.Linear(self.time_dim, self.inner_dim)

        for mod in self.out_modal:
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
            nn.Linear(4 * inner_dim, 2 * inner_dim),
            nn.ReLU(), 
            nn.Dropout(p=pooler_dropout),
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim, 1)
        )

    def get_mod_mask(self, pair):
        np_pair = pair.cpu().numpy()
        is_modal_a = self.is_modal[np_pair[:, 0, np.newaxis], :]
        is_modal_b = self.is_modal[np_pair[:, 1, np.newaxis], :]
        mask_a = (is_modal_a) >= 1
        mask_b = (is_modal_b) >= 1
        return torch.from_numpy(mask_a).to(pair.device).squeeze(), \
                torch.from_numpy(mask_b).to(pair.device).squeeze()

    def get_is_modal(self):
        # judge if self have is_modal
        is_modal = np.zeros([len(self.raw_data.keys()), len(self.out_modal)])
        for d in self.raw_data.keys():
            for mdl in self.out_modal:
                if self.raw_data[d][mdl] is not None:
                    is_modal[self.drug_dict[d], self.out_modal.index(mdl)] = 1
        self.is_modal = is_modal
        return is_modal

    def get_pred_batch(self, a, b, gene_exps, time_embeddings):
        if len(a.shape) == 2:
            a, b = a.unsqueeze(1), b.unsqueeze(1)
        gene_exps_batch = gene_exps.repeat(1, a.shape[1], 1)
        time_embeddings_batch = time_embeddings.repeat(1, a.shape[1], 1)
        x_hat = self.classifier_1(torch.cat([a, b, gene_exps_batch, time_embeddings_batch], dim=-1))
        pred = self.classifier_2(x_hat)
        return pred.squeeze(), x_hat.squeeze()
    
    def dict_to_emb(self, out_a_dict, out_b_dict):
        out_a = torch.stack([out_a_dict[mod] for mod in self.out_modal], dim=1)
        out_b = torch.stack([out_b_dict[mod] for mod in self.out_modal], dim=1)
        return out_a, out_b

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
    
    def encode_time(self, time):
        time_npy = time.cpu().numpy().squeeze()
        time_enc = []
        for t in time_npy:
            time_enc.append(self.time_emb[t, :].reshape(1, -1))
        time_enc = np.concatenate(time_enc, axis=0)
        time_enc = torch.from_numpy(time_enc).to(time.device)
        return time_enc

    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, gene_exps, time, pair=None, labels=None):

        gene_exps_neighbors = []
        for hop in range(self.n_hop):
            gene_exps_neighbors.append(torch.LongTensor([self.gene_exp_neighbor_set[c][hop] \
                                                       for c in gene_exps.squeeze(1).cpu().numpy().tolist()]).to(drug_a.device)) 
        gene_exp_neighbors_emb_list = self._get_neighbor_emb(gene_exps_neighbors)
        gene_exp_i_list = self._interaction_aggregation(gene_exp_neighbors_emb_list)
        gene_exp_embeddings = self._aggregation(gene_exp_i_list)
        gene_exp_embeddings = self.layernorm_gene_exp(gene_exp_embeddings)
        gene_exp_embeddings = gene_exp_embeddings.unsqueeze(1)

        time_enc = self.encode_time(time)
        # convert time enc dtype to gene_exp_embeddings dtype
        time_enc = time_enc.type(gene_exp_embeddings.dtype)
        time_embeddings = self.time_layer(time_enc).unsqueeze(1)

        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)

        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)

        # get the input of other modal
        other_modal_a, other_modal_b = self._get_pair_input(pair)
        out_a_dict, out_b_dict = self._get_pair_output(other_modal_a, other_modal_b)
        out_a_dict['SMILES'], out_a_dict['Graph'] = ta, ga
        out_b_dict['SMILES'], out_b_dict['Graph'] = tb, gb
        a_i_out, b_i_out = self.dict_to_emb(out_a_dict, out_b_dict)
        a_mask, b_mask = self.get_mod_mask(pair)
        a_exp_i, b_exp_i, a_exp_mask, b_exp_mask = self.expand_ab(a_i_out, b_i_out, a_mask, b_mask)
        mask  = torch.multiply(a_exp_mask, b_exp_mask)
        pred, all_combo_emb = self.get_pred_batch(a_exp_i, b_exp_i, gene_exp_embeddings, time_embeddings)
        if len(pred.shape) == 1: pred, x_emb = pred.unsqueeze(-1), x_emb.unsqueeze(1)

        out = self.wta_layer(pred, mask)

        combo_idxes = np.random.choice(range(all_combo_emb.size(1)), size=2, replace=False, \
                                       p=[1/all_combo_emb.size(1)] * all_combo_emb.size(1))
        assert combo_idxes[0] != combo_idxes[1]
        sub_out_1 = self.classifier_2(all_combo_emb[:, combo_idxes[0], :])
        sub_out_2 = self.classifier_2(all_combo_emb[:, combo_idxes[1], :])
        xc_1 = all_combo_emb[:, combo_idxes[0], :].squeeze()
        xc_2 = all_combo_emb[:, combo_idxes[1], :].squeeze()

        consine = self.get_cosine_loss(xc_1, xc_2)

        return out.reshape(-1, 1), consine, 0.5 * (sub_out_1 + sub_out_2).reshape(-1, 1)
    
    def get_embs(self, drug_a, dv_drug_a, drug_b, dv_drug_b, gene_exps, time, pair=None, labels=None):
        gene_exps_neighbors = []
        for hop in range(self.n_hop):
            gene_exps_neighbors.append(torch.LongTensor([self.gene_exp_neighbor_set[c][hop] \
                                                       for c in gene_exps.squeeze(1).cpu().numpy().tolist()]).to(drug_a.device)) 
        gene_exp_neighbors_emb_list = self._get_neighbor_emb(gene_exps_neighbors)
        gene_exp_i_list = self._interaction_aggregation(gene_exp_neighbors_emb_list)
        gene_exp_embeddings = self._aggregation(gene_exp_i_list)
        gene_exp_embeddings = self.layernorm_gene_exp(gene_exp_embeddings)
        gene_exp_embeddings = gene_exp_embeddings.unsqueeze(1)
        time_enc = self.encode_time(time)
        # convert time enc dtype to gene_exp_embeddings dtype
        time_enc = time_enc.type(gene_exp_embeddings.dtype)
        time_embeddings = self.time_layer(time_enc).unsqueeze(1)
        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)
        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)
        # get the input of other modal
        other_modal_a, other_modal_b = self._get_pair_input(pair)
        out_a_dict, out_b_dict = self._get_pair_output(other_modal_a, other_modal_b)
        out_a_dict['SMILES'], out_a_dict['Graph'] = ta, ga
        out_b_dict['SMILES'], out_b_dict['Graph'] = tb, gb
        a_i_out, b_i_out = self.dict_to_emb(out_a_dict, out_b_dict)
        a_mask, b_mask = self.get_mod_mask(pair)
        a_exp_i, b_exp_i, a_exp_mask, b_exp_mask = self.expand_ab(a_i_out, b_i_out, a_mask, b_mask)
        pred, all_combo_emb = self.get_pred_batch(a_exp_i, b_exp_i, gene_exp_embeddings, time_embeddings)
        if len(pred.shape) == 1: pred, x_emb = pred.unsqueeze(-1), x_emb.unsqueeze(1)
        indices = torch.topk(pred, k=self.topk, dim=-1)[1].squeeze()
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, all_combo_emb.size(2))
        topk_embeddings = all_combo_emb.gather(1, expanded_indices)
        mean_topk_embeddings = torch.mean(topk_embeddings, dim=1)
        return mean_topk_embeddings
    
    def get_cosine_loss(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = self.contrastive_loss(logits, targets)
        return loss

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
        for mod in self.out_modal:
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
        
        for mod in self.out_modal:
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
        
