from torch import layout, nn
import torch
from fairseq import utils
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import os
import networkx as nx
import collections
import pdb


class XenograftDataPPI():
    def __init__(self, 
                 aux_data_dir,
                 drug_target_path='baselines/GraphSynergy-master/data_ours_3fold/drug_protein.csv',
                 n_hop=2, 
                 n_memory=32,):
        self.aux_data_dir = aux_data_dir
        self.n_hop = n_hop
        self.n_memory = n_memory
        self.drug_target_path = drug_target_path

        self.ppi_df, self.cpi_df = self.load_data()

        self.node_map_dict, self.node_num_dict = self.get_node_map_dict()

        self.df_node_remap()

        self.graph = self.build_graph()

        self.model_protein_dict = self.get_target_dict()

        self.models = list(self.model_protein_dict.keys())

        self.model_neighbor_set = self.get_neighbor_set(items=self.models,
                                                       item_target_dict=self.model_protein_dict)

        self.drug_target = self.load_drug_target(drug_target_path=drug_target_path)
        self.drug_neighbor_set = self.get_neighbor_set(items=self.drug_target.keys(),
                                                       item_target_dict=self.drug_target)

    def get_c_neighbor_set(self):
        return self.model_neighbor_set
    
    def get_drug_neighbor_set(self):
        return self.drug_neighbor_set
    
    def get_node_num_dict(self):
        return self.node_num_dict
    
    def load_data(self):

        ppi_df = pd.read_excel(os.path.join(self.aux_data_dir, 'protein-protein_network.xlsx'))
        cpi_df = pd.read_csv(os.path.join(self.aux_data_dir, 'model_protein2.csv'))

        return ppi_df, cpi_df
    
    def load_drug_target(self, drug_target_path='data/drug_target.csv'):
        self.drug_target_csv = pd.read_csv(drug_target_path, sep='\t')
        self.drug_target = {}
        for i in self.drug_target_csv.index:
            drug, protein = self.drug_target_csv.loc[i, 'drug'], self.drug_target_csv.loc[i, 'protein']
            if drug not in self.drug_target:
                self.drug_target[drug] = [protein]
            else:
                self.drug_target[drug].append(protein)
        return self.drug_target
    
    def get_node_map_dict(self):
        protein_node = list(set(self.ppi_df['protein_a']) | set(self.ppi_df['protein_b']))
        model_fpkm_dir = os.path.dirname(self.drug_target_path)
        model_fpkm_path = os.path.join(model_fpkm_dir, 'model_fpkm.csv')
        model_fpkm = pd.read_csv(model_fpkm_path)
        model_node = model_fpkm['model_names'].tolist()


        node_num_dict = {'protein': len(protein_node), 'model': len(model_node)}
        mapping = {protein_node[idx]:idx for idx in range(len(protein_node))}
        mapping.update({model_node[idx]:idx for idx in range(len(model_node))})

        # display data info
        print('undirected graph')
        print('# proteins: {0}, # models: {1}'.format(
                len(protein_node), len(model_node)))
        print('# protein-protein interactions: {0}, # model-protein associations: {1}'.format(
            len(self.ppi_df), len(self.cpi_df)))

        return mapping, node_num_dict

    def df_node_remap(self):
        self.ppi_df['protein_a'] = self.ppi_df['protein_a'].map(self.node_map_dict)
        self.ppi_df['protein_b'] = self.ppi_df['protein_b'].map(self.node_map_dict)
        self.ppi_df = self.ppi_df[['protein_a', 'protein_b']]

        self.cpi_df['model'] = self.cpi_df['model'].map(self.node_map_dict)
        self.cpi_df['protein'] = self.cpi_df['protein'].map(self.node_map_dict)
        self.cpi_df = self.cpi_df[['model', 'protein']]


    def build_graph(self):
        tuples = [tuple(x) for x in self.ppi_df.values]
        graph = nx.Graph()
        graph.add_edges_from(tuples)
        return graph

    def get_target_dict(self):
        cp_dict = collections.defaultdict(list)
        model_list = list(set(self.cpi_df['model']))
        for model in model_list:
            model_df = self.cpi_df[self.cpi_df['model']==model]
            target = list(set(model_df['protein']))
            cp_dict[model] = target

        return cp_dict

    def get_neighbor_set(self, items, item_target_dict):
        print('constructing neighbor set ...')

        neighbor_set = collections.defaultdict(list)
        for item in items:
            for hop in range(self.n_hop):
                # use the target directly
                if hop == 0:
                    replace = len(item_target_dict[item]) < self.n_memory
                    target_list = list(np.random.choice(item_target_dict[item], size=self.n_memory, replace=replace))
                else:
                    # use the last one to find k+1 hop neighbors
                    origin_nodes = neighbor_set[item][-1]
                    neighbors = []
                    for node in origin_nodes:
                        neighbors += self.graph.neighbors(node)
                    # sample
                    replace = len(neighbors) < self.n_memory
                    target_list = list(np.random.choice(neighbors, size=self.n_memory, replace=replace))
                
                neighbor_set[item].append(target_list)

        return neighbor_set

