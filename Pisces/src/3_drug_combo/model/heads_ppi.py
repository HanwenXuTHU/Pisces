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


class GDSCDataPPI():
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

        self.cell_protein_dict = self.get_target_dict()

        self.cells = list(self.cell_protein_dict.keys())

        self.cell_neighbor_set = self.get_neighbor_set(items=self.cells,
                                                       item_target_dict=self.cell_protein_dict)

        self.drug_target = self.load_drug_target(drug_target_path=drug_target_path)
        self.drug_neighbor_set = self.get_neighbor_set(items=self.drug_target.keys(),
                                                       item_target_dict=self.drug_target)

    def get_c_neighbor_set(self):
        return self.cell_neighbor_set
    
    def get_drug_neighbor_set(self):
        return self.drug_neighbor_set
    
    def get_node_num_dict(self):
        return self.node_num_dict
    
    def load_data(self):

        ppi_df = pd.read_excel(os.path.join(self.aux_data_dir, 'protein-protein_network.xlsx'))
        cpi_df = pd.read_csv(os.path.join(self.aux_data_dir, 'cell_protein.csv'))

        return ppi_df, cpi_df
    
    def load_drug_target(self, drug_target_path='data/drug_target.csv'):
        self.drug_target_csv = pd.read_csv(drug_target_path)
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
        cell_tpm_dir = os.path.dirname(self.drug_target_path)
        cell_tpm_path = os.path.join(cell_tpm_dir, 'cell_tpm.csv')
        cell_tpm = pd.read_csv(cell_tpm_path, index_col=0)
        cell_node = cell_tpm['cell_line_names'].tolist()


        node_num_dict = {'protein': len(protein_node), 'cell': len(cell_node)}
        mapping = {protein_node[idx]:idx for idx in range(len(protein_node))}
        mapping.update({cell_node[idx]:idx for idx in range(len(cell_node))})

        # display data info
        print('undirected graph')
        print('# proteins: {0}, # cells: {1}'.format(
                len(protein_node), len(cell_node)))
        print('# protein-protein interactions: {0}, # cell-protein associations: {1}'.format(
            len(self.ppi_df), len(self.cpi_df)))

        return mapping, node_num_dict

    def df_node_remap(self):
        self.ppi_df['protein_a'] = self.ppi_df['protein_a'].map(self.node_map_dict)
        self.ppi_df['protein_b'] = self.ppi_df['protein_b'].map(self.node_map_dict)
        self.ppi_df = self.ppi_df[['protein_a', 'protein_b']]

        self.cpi_df['cell'] = self.cpi_df['cell'].map(self.node_map_dict)
        self.cpi_df['protein'] = self.cpi_df['protein'].map(self.node_map_dict)
        self.cpi_df = self.cpi_df[['cell', 'protein']]


    def build_graph(self):
        tuples = [tuple(x) for x in self.ppi_df.values]
        graph = nx.Graph()
        graph.add_edges_from(tuples)
        return graph

    def get_target_dict(self):
        cp_dict = collections.defaultdict(list)
        cell_list = list(set(self.cpi_df['cell']))
        for cell in cell_list:
            cell_df = self.cpi_df[self.cpi_df['cell']==cell]
            target = list(set(cell_df['protein']))
            cp_dict[cell] = target

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
