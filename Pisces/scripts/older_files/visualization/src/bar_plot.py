import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

import plot_settings_bar
import plot_utils

trans_DeepDDS = [
    [[0.6542, 0.3697, 0.3841, 0.3570]],
    [[0.6509, 0.3676, 0.3807, 0.3537]],
    [[0.6420, 0.3709, 0.3672, 0.3399]]]
trans_GraphSyn = [
    [[0.6493, 0.4314, 0.3878, 0.3642]],
    [[0.646, 0.4192, 0.377, 0.3536]],
    [[0.6534, 0.441, 0.403, 0.3802]]]
trans_PRODeepSyn = [
    [[0.6559, 0.3072, 0.3654, 0.3344]],
    [[0.6441, 0.3212, 0.3476, 0.3163]],
    [[0.6454, 0.3083, 0.3516, 0.3193]]]
trans_DeepSynergy = [
    [[0.5236, 0.0607, 0.0981, 0.0247]],
    [[0.5127, 0.1838, 0.0507, 0.0462]],
    [[0.5099, 0.2033, 0.0395, 0.0364]]]

trans_AuDNNSyn = [
    [[0.5086, 0.194, 0.0353, 0.0318]],
    [[0.5491, 0.2945, 0.1715, 0.156]],
    [[0.5057, 0.2552, 0.0236, 0.0211]],
]
trans_ours = [
    [[0.7101, 0.4354, 0.4514, 0.4224]],
    [[0.6906, 0.4269, 0.4257, 0.3960]],
    [[0.7039, 0.4391, 0.4576, 0.4294]]]
trans_multi_modal = [
    [[0.7408, 0.4619, 0.4636, 0.4317]],
    [[0.7179, 0.4310, 0.4484, 0.4172]],
    [[0.7401, 0.4591, 0.4781, 0.4470]]]

ours_mean = np.mean(trans_ours, axis=0)
multi_mean = np.mean(trans_multi_modal, axis=0)
models_trans = ['DeepSynergy', 'AuDNNsynergy', 'PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Pisces', 'Pisces (multi-modal)']
trans = (trans_DeepSynergy, trans_AuDNNSyn, trans_PRODeepSyn, trans_GraphSyn, trans_DeepDDS, trans_ours, trans_multi_modal)

lc_DeepDDS = [
    [[0.6274, 0.2428, 0.3075, 0.2754]],
    [[0.5821, 0.1881, 0.2285, 0.1975]],
    [[0.5976, 0.1916, 0.2425, 0.2097]]
]

lc_PRODeepSyn = [
    [[0.7081, 0.2455, 0.3028, 0.2518]],
    [[0.5528, 0.1628, 0.172, 0.1483]],
    [[0.5904, 0.197, 0.2395, 0.2101]]
]

lc_DeepSynergy = [
    [[0.5097, 0.0799, 0.0402, 0.0351]],
    [[0.5, 0.0796, 0.0, 0]],
    [[0.505, 0.0694, 0.0224, 0.01816]]
]

lc_AuDnnSyn = [
    [[0.5591, 0.2316, 0.1888, 0.166]],
    [[0.542, 0.1835, 0.1465, 0.1279]],
    [[0.5284, 0.1837, 0.1055, 0.0927]]
]

lc_Ours = [
    [[0.6507, 0.2224, 0.3034, 0.2629]],
    [[0.6267, 0.2044, 0.2793, 0.2395]],
    [[0.6322, 0.2263, 0.2827, 0.2464]]
]

lc_multi_modal = [
    [[0.6618, 0.2311, 0.3157, 0.2751]],
    [[0.6506, 0.2205, 0.3, 0.2578]],
    [[0.6685, 0.2106, 0.2845, 0.2401]]
]
# new methods
'''lc_multi_modal = [
    [[0.6858, 0.2199, 0.2897, 0.2393]],
    [[0.6479, 0.1873, 0.2627, 0.2124]],
    [[0.6685, 0.1939, 0.2611, 0.2127]]
]'''

lc_GraphSyn = [
    [[0, 0, 0, 0]],
    [[0, 0, 0, 0]],
    [[0, 0, 0, 0]],
]

models_lc = ['DeepSynergy', 'AuDNNsynergy', 'PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Pisces', 'Pisces (multi-modal)']
lc = (lc_DeepSynergy, lc_AuDnnSyn, lc_PRODeepSyn, lc_GraphSyn, lc_DeepDDS, lc_Ours, lc_multi_modal)

ldc_DeepDDS = [
    [[0.5966, 0.2478, 0.2688, 0.2404]],
    [[0.6114, 0.259, 0.2905, 0.2647]],
    [[0.6089, 0.3085, 0.303, 0.2798]]
]

ldc_PRODeepSyn = [
    [[0.6119, 0.245, 0.2927, 0.2622]],
    [[0.6163, 0.2405, 0.288, 0.2596]],
    [[0.6121, 0.2328, 0.2868, 0.2579]]
]

ldc_DeepSynergy = [
    [[0.5075, 0.1836, 0.0305, 0.0279]],
    [[0.5077, 0.1293, 0.0319, 0.0283]],
    [[0.5087, 0.1669, 0.0361, 0.0315]]
]

ldc_AuDNNSyn = [
    [[0.5279, 0.2244, 0.1041, 0.0911]],
    [[0.5231, 0.2211, 0.0884, 0.0824]],
    [[0.5281, 0.2113, 0.105, 0.0929]]
]

ldc_GraphSyn = [
    [[0.6391, 0.2704, 0.3142, 0.2784]],
    [[0.6359, 0.2947, 0.313, 0.2846]],
    [[0.6259, 0.3143, 0.3119, 0.2843]]
]

ldc_Ours = [
    [[0.6609, 0.3085, 0.3569, 0.3227]],
    [[0.6081, 0.2253, 0.2675, 0.2376]],
    [[0.6501, 0.3509, 0.3728, 0.3477]]
]

ldc_multi_modal = [
    [[0.6950, 0.3215, 0.3867, 0.3504]],
    [[0.6949, 0.3450, 0.3855, 0.3546]],
    [[0.6908, 0.3734, 0.4073, 0.3783]]
]

models_ldc = ['DeepSynergy', 'AuDNNsynergy', 'PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Pisces', 'Pisces (multi-modal)']
ldc = (ldc_DeepSynergy, ldc_AuDNNSyn, ldc_PRODeepSyn, ldc_GraphSyn, ldc_DeepDDS, ldc_Ours, ldc_multi_modal)


settings = ['Vanilla CV', 'Stratified CV for\ndrug combinations', 'Stratified CV for\ncell lines']



def bar_plot(dataset, models, labels, name, metrics_id, y_lim=None):

    
    ax = plot_settings_bar.get_wider_axis(3, 4)

    colors = [plot_settings_bar.get_model_colors(mod) for mod in models]
    
    means = []
    stderrs = []

    for data in dataset:
        task_vals = []
        for result in data:
            result = np.concatenate(result)
            task_vals.append((np.mean(result[:, metrics_id]), np.std(result[:, metrics_id]) / np.sqrt(result.shape[0])))
        means.append([v[0] for v in task_vals])
        stderrs.append([v[1] for v in task_vals])

    min_val = [0.45, 0.0, 0, 0]


    plot_utils.grouped_barplot_graphsyn(
        ax, means, 
        settings,
        xlabel='', ylabel=name, color_legend=labels,
        nested_color=colors, nested_errs=stderrs, tickloc_top=False, rotangle=45, anchorpoint='right',
        min_val=min_val[metrics_id], y_lim=y_lim)
    
    plot_utils.format_ax(ax)

    
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
                            ncols=1)
    
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.8, 1.01))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    

    # plt.savefig(f'bar_{name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'bar_{name}.png', dpi=300, bbox_inches='tight')

dataset = (trans, ldc, lc)

bar_plot(dataset, models_trans, models_trans, 'BACC', 0)
bar_plot(dataset, models_trans, models_trans, 'AUPRC', 1)
bar_plot(dataset, models_trans, models_trans, r'$F_1$', 2)
bar_plot(dataset, models_trans, models_trans, 'KAPPA', 3, (0, 0.52))