#!/usr/bin/env python
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from STDISTAL.STDISTAL import run_STDISTAL


paths = { # seqFISH+ 
    
    'sc_path': './data/sc_data',
    'ST_path': './data/ST_data',
    'output_path': './output',
}


# paths = {
#     'sc_path': './seqfish/sc_data',
#     'ST_path': './seqfish/ST_data',
#     'output_path': './output',
# }


# paths = {
#     'sc_path': './merfish/sc_data',
#     'ST_path': './merfish/ST_data',
#     'output_path': './output',
# }


find_marker_genes_paras = {
    'preprocess': True,
    'normalize': True,
    'log': True,
    'highly_variable_genes': False,
    'highly_variable_gene_num': None,
    'regress_out': False,
    'PCA_components': 30, 
    'marker_gene_method': 'logreg',
    'top_gene_per_type': 100,
    'filter_wilcoxon_marker_genes': True,
    'pvals_adj_threshold': 0.10,
    'log_fold_change_threshold': 1,
    'min_within_group_fraction_threshold': None,
    'max_between_group_fraction_threshold': None,
}


pseudo_spot_simulation_paras = {
    'spot_num': 30000, # 30000,
    'min_cell_num_in_spot': 8,
    'max_cell_num_in_spot': 12,
    'generation_method': 'celltype',
    'max_cell_types_in_spot': 4,   
}

data_normalization_paras = {
    'normalize': True, 
    'log': True, 
    'scale': False,
}


integration_for_adj_paras = {
    'batch_removal_method': None, 
    'dim': 30, 
    'dimensionality_reduction_method': 'PCA',
    'scale': True,
}



inter_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine', 
    'corr_dist_neighbors': 20, 
}
real_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine',  
    'corr_dist_neighbors': 10,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}
pseudo_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine', 
    'corr_dist_neighbors': 20,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}



spatial_adj_paras = {
    'link_method': 'soft', 
    'space_dist_threshold': 2,
}



integration_for_feature_paras = {
    'batch_removal_method': None, 
    'dimensionality_reduction_method': None, 
    'dim': 80,
    'scale': True,
}



GCN_paras = {
    'epoch_n': 300,
    'dim': 80,
    'common_hid_layers_num': 1,
    'fcnn_hid_layers_num': 1,
    'dropout': 0,
    'learning_rate_SGD': 2e-1,
    'weight_decay_SGD': 3e-4,
    'momentum': 0.9,
    'dampening': 0,
    'nesterov': True,
    'early_stopping_patience': 20, 
    'clip_grad_max_norm': 1,
    'print_loss_epoch_step': 20,
}




if __name__ == "__main__":

    results =  run_STDISTAL(paths,
                        load_test_groundtruth = False,
                        use_marker_genes = True,
                        external_genes = False,
                        find_marker_genes_paras = find_marker_genes_paras,
                        generate_new_pseudo_spots = True, 
                        pseudo_spot_simulation_paras = pseudo_spot_simulation_paras,
                        data_normalization_paras = data_normalization_paras,
                        integration_for_adj_paras = integration_for_adj_paras,
                        inter_exp_adj_paras = inter_exp_adj_paras,
                        spatial_adj_paras = spatial_adj_paras,
                        real_intra_exp_adj_paras = real_intra_exp_adj_paras,
                        pseudo_intra_exp_adj_paras = pseudo_intra_exp_adj_paras,
                        integration_for_feature_paras = integration_for_feature_paras,
                        GCN_paras = GCN_paras,
                        fraction_pie_plot = True,
                        cell_type_distribution_plot = True,
                        n_jobs = -1, 
                        GCN_device = 'GPU'
                        )

    results.write_h5ad(paths['output_path']+'/results.h5ad')
