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


class HeadRandomPair(nn.Module):
    
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
                 fp16=True,
                 mix=False,
                 is_bitop=False,
                 all_modal=True):

        super().__init__()
        
        self.num_classes = num_classes
        self.inner_dim = inner_dim
        self.emb_dim = inner_dim
        self.n_hop = 2
        self.n_memory = n_memory
        self.topk = topk
        self.fp16 = fp16
        self.mix = mix
        self.is_bitop = is_bitop

        ppi_loader = DataPPI(
            aux_data_dir='data/ppi',
            drug_target_path=drug_target_path,
            n_hop=self.n_hop,
            n_memory=self.n_memory)

        self.seen_drugs = seen_drugs
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
        nn.init.xavier_uniform_(self.rel_emb.weight)
        self.aggregation_function = nn.Linear(self.emb_dim * 2 * self.n_hop, self.emb_dim)

        self.transformer_proj_head = nn.Linear(input_dim, inner_dim)
        self.graph_proj_head = nn.Linear(dv_input_dim, inner_dim)
        self.layernorm_drug = nn.LayerNorm(inner_dim)
        self.layernorm_rel = nn.LayerNorm(inner_dim)

        if mix:
            self.mix_layer = nn.MultiheadAttention(2 * inner_dim, 4, dropout=pooler_dropout, batch_first=True)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(pooler_dropout)

        self.cst_modal = ['SMILES', 'Graph', '3D', 'Side effect', 'Drug Sensitivity (NCI60)', \
                          'Text', 'Drug Ontology', 'Drug target']
        if all_modal:
            self.out_modal = ['SMILES', 'Graph', '3D', 'Side effect', 'Drug Sensitivity (NCI60)', \
                            'Text', 'Drug Ontology', 'Drug target']
        else:
            self.out_modal = ['SMILES', 'Graph', '3D', 'Text']
        self.raw_data_path = raw_data_path
        self.drug_dict = drug_dict
        self.id2drug = {v: k for k, v in self.drug_dict.items()}
        self.drug_n = len(self.drug_dict)
        # load raw data
        with open(self.raw_data_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        for mod in self.cst_modal:
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
            nn.Linear(2 * inner_dim, 2 * inner_dim),
            nn.Tanh(), 
            nn.Dropout(p=pooler_dropout),
        )

        self.rels_linear = nn.Sequential(
            nn.Linear(3*self.emb_dim, inner_dim),
            nn.Tanh(),
            nn.Dropout(p=pooler_dropout),
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.Tanh(), 
            nn.Dropout(p=pooler_dropout),
        )

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def get_pred_batch(self, a, b, rels):
        if len(a.shape) == 2:
            a, b = a.unsqueeze(1), b.unsqueeze(1)
        rels_batch = rels.repeat(1, a.shape[1], 1)
        x = torch.cat([a, b], dim=-1)
        x = self.dropout(x)
        x = self.classifier_1(x)
        return self.x_rels_pred(x, rels_batch)
    
    def x_rels_pred(self, x, rels_batch):
        input = torch.cat([x, rels_batch], dim=-1)
        del_rel = self.rels_linear(input)
        rel_hat = rels_batch + del_rel
        if self.mix:
            x_mix = self.activation_fn(self.mix_layer(x, x, x)[0])
            x_hat = x_mix + x
            x_hat = self.dropout(x_hat)
        else:
            x_hat = x
        x_hat = self.classifier_2(x_hat)
        pred = torch.multiply(rel_hat, x_hat).sum(dim=-1)
        return pred.squeeze(), x.squeeze()
    
    def dict_to_emb(self, out_a_dict, out_b_dict, modal_list):
        out_a = torch.stack([out_a_dict[mod] for mod in modal_list], dim=1)
        out_b = torch.stack([out_b_dict[mod] for mod in modal_list], dim=1)
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

    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, net_rel, pair=None, 
                random_a=None, dv_random_a=None, random_b=None, dv_random_b=None, random_pair=None,
                labels=None):

        rels = self.rel_emb(net_rel)
        # create random int tensors with the same shape of rels
        random_rels = torch.randint_like(net_rel, self.num_classes).to(rels.device)
        random_rels_emb = self.rel_emb(random_rels)

        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)

        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)

        # get the input of other modal
        other_modal_a, other_modal_b = self._get_pair_input(pair)
        out_a_dict, out_b_dict = self._get_pair_output(other_modal_a, other_modal_b)
        out_a_dict['SMILES'], out_a_dict['Graph'] = ta, ga
        out_b_dict['SMILES'], out_b_dict['Graph'] = tb, gb
        a_i_out, b_i_out = self.dict_to_emb(out_a_dict, out_b_dict, self.out_modal)
        a_exp_out, b_exp_out = self.expand_ab(a_i_out, b_i_out)
        if self.cst_modal == self.out_modal:
            pred, all_combo_emb = self.get_pred_batch(a_exp_out, b_exp_out, rels)
        else:
            pred, _ = self.get_pred_batch(a_exp_out, b_exp_out, rels)

        if self.is_bitop:
            out_large, _ = torch.topk(pred, k=self.topk, largest=True, dim=1)
            out_small, _ = torch.topk(pred, k=self.topk, largest=False, dim=1)
            out = torch.cat([out_large, out_small], dim=1)
        else:
            out, _ = torch.topk(pred, k=self.topk, largest=True, dim=1)
        out = torch.mean(out, dim=1, keepdim=True)

        if self.cst_modal != self.out_modal:
            a_i_cst, b_i_cst = self.dict_to_emb(out_a_dict, out_b_dict, self.cst_modal)
            a_exp_cst, b_exp_cst = self.expand_ab(a_i_cst, b_i_cst)
            _, all_combo_emb = self.get_pred_batch(a_exp_cst, b_exp_cst, rels)

        combo_idxes = np.random.choice(range(all_combo_emb.size(1)), size=2, replace=False, \
                                       p=[1/all_combo_emb.size(1)] * all_combo_emb.size(1))
        assert combo_idxes[0] != combo_idxes[1]
        sub_out_1, xc_1 = self.x_rels_pred(all_combo_emb[:, combo_idxes[0], :].unsqueeze(1), rels)
        sub_out_2, xc_2 = self.x_rels_pred(all_combo_emb[:, combo_idxes[1], :].unsqueeze(1), rels)

        consine = self.get_cosine_loss(xc_1, xc_2)

        #add addition loss on all drugs
        # randomly decide if the following codes are executed
        if random_a is not None:
            tra = self.transformer_proj_head(random_a)
            trb = self.transformer_proj_head(random_b)

            gra = self.graph_proj_head(dv_random_a)
            grb = self.graph_proj_head(dv_random_b)
            random_modal_a, random_modal_b = self._get_pair_input(random_pair)
            random_a_dict, random_b_dict = self._get_pair_output(random_modal_a, random_modal_b)
            random_a_dict['SMILES'], random_a_dict['Graph'] = tra, gra
            random_b_dict['SMILES'], random_b_dict['Graph'] = trb, grb
            ra_i_out, rb_i_out = self.dict_to_emb(random_a_dict, random_b_dict, self.out_modal)
            # randomly order ra_i_out and rb_i_out
            ra_i_out = ra_i_out[torch.randperm(ra_i_out.size(0))]
            rb_i_out = rb_i_out[torch.randperm(rb_i_out.size(0))]
            ra_exp_out, rb_exp_out = self.expand_ab(ra_i_out, rb_i_out)
            _, random_combo_emb = self.get_pred_batch(ra_exp_out, rb_exp_out, random_rels_emb)

            random_combo_idxes = np.random.choice(range(random_combo_emb.size(1)), size=2, replace=False, \
                                            p=[1/random_combo_emb.size(1)] * random_combo_emb.size(1))
            assert random_combo_idxes[0] != random_combo_idxes[1]
            _, random_xc_1 = self.x_rels_pred(random_combo_emb[:, random_combo_idxes[0], :].unsqueeze(1), random_rels_emb)
            _, random_xc_2 = self.x_rels_pred(random_combo_emb[:, random_combo_idxes[1], :].unsqueeze(1), random_rels_emb)

            xc_1 = torch.cat([xc_1, random_xc_1], dim=0)
            xc_2 = torch.cat([xc_2, random_xc_2], dim=0)

            random_consine = self.get_cosine_loss(xc_1, xc_2)
            return out.reshape(-1, 1), random_consine, 0.5 * (sub_out_1 + sub_out_2).reshape(-1, 1)
        else:
            return out.reshape(-1, 1), consine, 0.5 * (sub_out_1 + sub_out_2).reshape(-1, 1)
    
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
        for mod in self.cst_modal:
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
        out_a, out_b = {}, {}
        for mod in self.cst_modal:
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
        
