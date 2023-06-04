import sys
import pickle
import numpy as np
import seaborn as sns
sys.path.append('/homes/gws/xuhw/research_projects/Pisces/')

from fig_utils import *


# define a function that load the pickle file
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def plot_overlap_fig(save_path, drug_features, cmap):
    extension_processed_path = '/homes/gws/xuhw/research_projects/Pisces/extension_data/process_by_inchi/'
    overlap_matrix, feature_lists = calculate_dataset_overlap(extension_processed_path, drug_features)
    overlap_matrix = overlap_matrix.astype(np.float32)
    # normalize overlap_matrix for each row
    for i in range(overlap_matrix.shape[0]):
        overlap_matrix[i, :] = overlap_matrix[i, :] / overlap_matrix[i, i]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.6, FIG_HEIGHT*1.6))
    im = heatmap(100*overlap_matrix, feature_lists, feature_lists, ax=ax,
                    cmap=cmap, vmax=120)
    texts = annotate_heatmap(im, valfmt="{x:.2f}%")
    ax.set_title('Features overlap', fontdict=csfont)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_twosides_modal_distribution(path='/homes/gws/xuhw/research_projects/Pisces/data/two_sides/drug_modalities_twosides.pkl'):
    drug_modalities = load_obj(path)
    N = len(drug_modalities)
    modality_names = drug_modalities[list(drug_modalities.keys())[0]].keys()
    one_hot = np.zeros((N, len(modality_names)))
    for i, drug in enumerate(drug_modalities):
        for j, modality in enumerate(modality_names):
            if drug_modalities[drug][modality] is not None:
                one_hot[i, j] = 1
    for j, modality in enumerate(modality_names):
        print(modality, np.sum(one_hot[:, j])/len(drug_modalities))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.2, FIG_HEIGHT*2))
    # cusomize the qualitative color map for value 0 and 1
    cmap = mpl.colors.ListedColormap(['#99d594', '#fc8d59'])
    #cmap = LinearSegmentedColormap.from_list('mycmap', ['#99d594', '#fc8d59'])
    sns.heatmap(one_hot, cmap=cmap)
    ax.set_xticklabels(modality_names, rotation=45, ha='right')
    # set the color bar tick labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['False', 'True'])
    ax.set_ylabel('Drugs')
    ax.set_xlabel('Modalities')
    ax.set_aspect(2*one_hot.shape[1] / one_hot.shape[0])
    fig.tight_layout()
    plt.savefig('/homes/gws/xuhw/research_projects/Pisces/figures/save_figs/twosides_modalities.pdf')


def plot_drugbank_modal_distribution(path='/homes/gws/xuhw/research_projects/Pisces/data/drugbank_ddi/drug_modalities.pkl'):
    drug_modalities = load_obj(path)
    N = len(drug_modalities)
    modality_names = drug_modalities[list(drug_modalities.keys())[0]].keys()
    one_hot = np.zeros((N, len(modality_names)))
    for i, drug in enumerate(drug_modalities):
        for j, modality in enumerate(modality_names):
            if drug_modalities[drug][modality] is not None:
                one_hot[i, j] = 1
    for j, modality in enumerate(modality_names):
        print(modality, np.sum(one_hot[:, j])/len(drug_modalities))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH*1.2, FIG_HEIGHT*2))
    # cusomize the qualitative color map for value 0 and 1
    cmap = mpl.colors.ListedColormap(['#99d594', '#fc8d59'])
    #cmap = LinearSegmentedColormap.from_list('mycmap', ['#99d594', '#fc8d59'])
    sns.heatmap(one_hot, cmap=cmap)
    ax.set_xticklabels(modality_names, rotation=45, ha='right')
    # set the color bar tick labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['False', 'True'])
    ax.set_ylabel('Drugs')
    ax.set_xlabel('Modalities')
    ax.set_aspect(2*one_hot.shape[1] / one_hot.shape[0])
    fig.tight_layout()
    plt.savefig('/homes/gws/xuhw/research_projects/Pisces/figures/save_figs/drugbank_modalities.pdf')

plot_drugbank_modal_distribution()
