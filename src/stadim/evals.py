import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from collections import Counter
import scipy.sparse as sp
from typing import List, Dict, Optional, Literal
import warnings
import hnswlib

def calculate_tau_index(expression, labels):
    region_means = []
    for r in np.unique(labels):
        region_means.append(np.mean(expression[labels == r]))
    region_means = np.array(region_means)
    
    if region_means.min() < 0:
        region_means = region_means - region_means.min()

    if np.max(region_means) > 0:
        x_hat = region_means / np.max(region_means)
        tau = np.sum(1 - x_hat) / (len(region_means) - 1)
        return np.clip(tau, 0, 1)
    else:
        return 0.0

def calculate_fold_change(target_expression, background_expression):
    fc = np.mean(target_expression) / (np.mean(background_expression) + 1e-10)
    return fc

def calculate_cohens_d(target_expression, background_expression):
    target = np.array(target_expression)
    background = np.array(background_expression)

    n1 = len(target)
    n2 = len(background)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1 = np.mean(target)
    mean2 = np.mean(background)
    var1 = np.var(target, ddof=1)  # ddof=1 means /n-1
    var2 = np.var(background, ddof=1)
    
    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_variance)
    
    if pooled_std == 0:
        return 0.0

    cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d

def calculate_new_morani(adata, layers, gene_region_dict):
    from scipy import sparse
    results = []
    
    if 'connectivities' not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=6, use_rep='spatial')
    global_w = adata.obsp['connectivities']
    labels = adata.obs['Label'].values

    for layer in layers:
        if layer not in adata.layers: continue
    
        for gene, target_region in gene_region_dict.items():
            if gene not in adata.var_names: continue
            
            vec = adata[:, gene].layers[layer]
            if sparse.issparse(vec): vec = vec.toarray()
            vec = vec.flatten()
            if np.var(vec) == 0:
                print(f" Method: {layer:10} | Gene: {gene:10} | Warning: Constant expression, skipping.")
                continue
                
            # --- 1. 全局 Moran's I ---
            # gm = sc.metrics.morans_i(global_w, vals=adata[:, gene].layers[layer].T)
            gm = sc.metrics.morans_i(global_w, vals=vec)
            gm_val = gm[0] if hasattr(gm, '__len__') else gm
            
            # --- 2. 计算背景 Moran's I ---
            bg_mask = (labels != target_region)
            
            if bg_mask.sum() > 10:
                bg_vec = vec[bg_mask]
                
                # 如果背景区域表达量为常数，视为完全随机分布 (I=0)
                if np.var(bg_vec) == 0:
                    bg_moran_val = 0.0
                else:
                    # 切片获取背景区域的子图邻接矩阵
                    bg_w = global_w[bg_mask, :][:, bg_mask]
                    bg_moran = sc.metrics.morans_i(bg_w, vals=bg_vec)
                    bg_moran_val = bg_moran[0] if hasattr(bg_moran, '__len__') else bg_moran
            else:
                bg_moran_val = 0.0 # 样本量太少视为无显著模式
            
            # --- 3. 计算 newI 指标 (gap) ---
            # 公式: newI = (I_global + (1 - |I_background|)) / 2
            new_i = (gm_val + (1 - abs(bg_moran_val))) / 2
            
            results.append({
                'Method': layer,
                'Gene': gene,
                'Region': target_region,
                'Global_Moran': gm_val,
                'Background_Moran': bg_moran_val,
                'newI': new_i
            })
            
            print(f"Method: {layer:12} | Gene: {gene:10} | newI: {new_i:.4f} (Global: {gm_val:.4f}, BG: {bg_moran_val:.4f})")

    return pd.DataFrame(results)
    
def calculate_sns_score(sdata, test_layers, group_key, batch_key='sample', donor_id=None, raw_layer='Raw (Norm)', n_subsample=None, n_hvg=2000, seed=None):
    if 'highly_variable' not in sdata.var:
        print(f"Warning: 'highly_variable' not in sdata.var. Calculating HVGs...")
        sc.pp.highly_variable_genes(sdata, n_top_genes=n_hvg, flavor='seurat_v3', layer='X_raw')
    
    hvg_indices = np.where(sdata.var['highly_variable'])[0]

    rng = np.random.default_rng(seed)
    if n_subsample and n_subsample < sdata.shape[0]:
        cell_indices = rng.choice(sdata.shape[0], size=n_subsample, replace=False)
    else:
        cell_indices = np.arange(sdata.shape[0])

    meta_df = sdata.obs.iloc[cell_indices].copy()
    meta_df['local_idx'] = np.arange(len(meta_df))

    def get_sub_matrix(layer_name):
        mat = sdata.layers[layer_name][cell_indices, :][:, hvg_indices]
        return mat.toarray() if hasattr(mat, "toarray") else mat

    results = []
    raw_sub = get_sub_matrix(raw_layer)
    all_layers = [raw_layer] + [l for l in test_layers if l != raw_layer]
    
    for layer in all_layers:
        if layer not in sdata.layers: 
            warnings.warn(f"Layer {layer} not found. Skipping.")
            continue
            
        # print(f"Evaluating: {layer}")
        den_sub = get_sub_matrix(layer)

        batch_cms, batch_crs, batch_gms, batch_grs = [], [], [], []
        batch_inter_cvs, batch_weights = [], []
        intra_cv_list = []

        for b, b_df in meta_df.groupby(batch_key):
            local_idx_b = b_df['local_idx'].values
            n_cells_b = len(local_idx_b)
            
            if n_cells_b < 2: 
                continue

            raw_sub_b = raw_sub[local_idx_b, :]
            den_sub_b = den_sub[local_idx_b, :]

            # --- 1. Structure Calculation (Batch-Aware) ---
            # Cell-level structure
            c_raw = np.nan_to_num(np.corrcoef(raw_sub_b))
            c_den = np.nan_to_num(np.corrcoef(den_sub_b))
            tri_c = np.triu_indices(n_cells_b, k=1)
            try:
                cms = max(0, 1 - cosine(c_raw[tri_c], c_den[tri_c]))
                crs = max(0, spearmanr(c_raw[tri_c], c_den[tri_c])[0])
            except:
                cms, crs = 0, 0

            # Gene-level structure
            g_raw = np.nan_to_num(np.corrcoef(raw_sub_b.T))
            g_den = np.nan_to_num(np.corrcoef(den_sub_b.T))
            tri_g = np.triu_indices(g_raw.shape[0], k=1)
            try:
                gms = max(0, 1 - cosine(g_raw[tri_g], g_den[tri_g]))
                grs = max(0, spearmanr(g_raw[tri_g], g_den[tri_g])[0])
            except:
                gms, grs = 0, 0

            batch_cms.append(cms)
            batch_crs.append(crs)
            batch_gms.append(gms)
            batch_grs.append(grs)

            # --- 2. CV Calculation: Inter-group (Batch-Aware) ---
            unique_labels_in_batch = b_df[group_key].unique()
            if len(unique_labels_in_batch) >= 2:
                group_means = []
                for l in unique_labels_in_batch:
                    idx_label = b_df[b_df[group_key] == l]['local_idx'].values
                    group_mat = den_sub[idx_label, :]
                    group_means.append(np.mean(group_mat, axis=0))
                
                means_mat = np.array(group_means)
                inter_cv_b = np.mean(np.std(means_mat, axis=0) / (np.mean(means_mat, axis=0) + 1e-10))
                batch_inter_cvs.append(inter_cv_b)
                batch_weights.append(n_cells_b)
                
        if sum(batch_weights) > 0:
            final_cms = np.average(batch_cms, weights=batch_weights)
            final_crs = np.average(batch_crs, weights=batch_weights)
            final_gms = np.average(batch_gms, weights=batch_weights)
            final_grs = np.average(batch_grs, weights=batch_weights)
        else:
            final_cms = final_crs = final_gms = final_grs = 0

        structure_score = (final_cms + final_crs + final_gms + final_grs) / 4

        # --- 2. CV Calculation: Intra-group (Batch-Aware) ---
        for (b, l), group in meta_df.groupby([batch_key, group_key]):
            idx_group = group['local_idx'].values
            if len(idx_group) < 2: 
                continue
            
            group_mat = den_sub[idx_group, :]
            m = np.mean(group_mat, axis=0) + 1e-10
            s = np.std(group_mat, axis=0)
            intra_cv_list.append(np.mean(s / m))
            
        cv_intra = np.mean(intra_cv_list) if intra_cv_list else 0
        cv_inter = np.average(batch_inter_cvs, weights=batch_weights) if batch_inter_cvs else 0

        results.append({
            'Method': layer, 
            'Sample': sdata.obs.get('sample', pd.Series(['N/A'])).iloc[0], 
            'Donor_ID': donor_id,
            'CMS': final_cms, 'CRS': final_crs, 'GMS': final_gms, 'GRS': final_grs,
            'Structure': structure_score, 
            'CV_intra': cv_intra, 
            'CV_inter': cv_inter
        })

    # --- 3. SNS score calculation ---
    df = pd.DataFrame(results)
    raw_info = df[df['Method'] == raw_layer].iloc[0]
    
    raw_cv_intra = raw_info['CV_intra'] if raw_info['CV_intra'] > 0 else 1e-10
    raw_cv_inter = raw_info['CV_inter'] if raw_info['CV_inter'] > 0 else 1e-10

    df['Noise'] = df['CV_intra'].apply(lambda x: max(0, 1 - (x / raw_cv_intra)))
    df['Signal'] = df['CV_inter'].apply(lambda x: np.exp(-np.abs(x / raw_cv_inter - 1)))
    df['SNS_Score'] = df['Structure'] * df['Noise'] * df['Signal']

    return df.sort_values('SNS_Score', ascending=False).reset_index(drop=True)

def calculate_batch_entropy(sdata, test_layers, k_range, group_key, donor_id=None, batch_key='sample', n_hvg=2000):

    n_obs = sdata.n_obs
    
    batch_labels = sdata.obs[batch_key].values
    region_labels = sdata.obs[group_key].values
    unique_batches = sdata.obs[batch_key].unique()
    n_batches = len(unique_batches)
    
    if 'highly_variable' not in sdata.var:
        sc.pp.highly_variable_genes(sdata, n_top_genes=n_hvg, flavor='seurat_v3')
    hvg_mask = sdata.var['highly_variable'].values
    
    max_k = max(k_range)
    avg_results = []

    for layer in test_layers:
        if layer not in sdata.layers: continue

        mat = sdata.layers[layer][:, hvg_mask]
        expr_matrix = mat.toarray() if hasattr(mat, "toarray") else mat
        
        dim = expr_matrix.shape[1]
        num_elements = expr_matrix.shape[0]

        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16, random_seed=2025)
        index.add_items(expr_matrix)
        index.set_ef(50)
        index.set_num_threads(1)
        
        ind, _ = index.knn_query(expr_matrix, k=max_k + 1)
        neighbor_idx_matrix = np.empty((n_obs, max_k), dtype=int)
        for i in range(n_obs):
            if ind[i, 0] == i:
                neighbor_idx_matrix[i] = ind[i, 1:max_k + 1] 
            else:
                neighbor_idx_matrix[i] = ind[i, :max_k]

        for k in k_range:
            current_neighbors = neighbor_idx_matrix[:, :k]            
            neighbor_regions = region_labels[current_neighbors] # (N,k)
            region_matches = (neighbor_regions == region_labels[:, None]) # (N,1)
            avg_region_ratio = np.mean(region_matches.sum(axis=1) / k)

            neighbor_batches = batch_labels[current_neighbors]
            
            def compute_row_entropy(row_labels):
                counts = Counter(row_labels)
                props = np.array(list(counts.values())) / k
                return -np.sum(props * np.log2(props))

            entropies = np.array([compute_row_entropy(nb) for nb in neighbor_batches])
            norm_entropy = entropies / np.log2(n_batches) if n_batches > 1 else 0
            
            avg_results.append({
                'Method': layer, 'Donor_ID': donor_id,
                'K': k,
                'Avg_Entropy': np.mean(norm_entropy),
                'Avg_Region_Ratio': avg_region_ratio
            })
            
    return pd.DataFrame(avg_results)
    
def calculate_cv(adata: 'anndata.AnnData', batch_key: str, label_key: str, methods: List[str], gene_type: Literal["HVGs", "LVGs", "ALL"]) -> pd.DataFrame:

    sedr_method_in_list = any('SEDR_hvg' in m for m in methods)
    use_sedr_genes = sedr_method_in_list and ('SEDR_hvg' in adata.layers) 

    hvg_candidates = []
    if use_sedr_genes and 'SEDR_hvg_genes' in adata.uns: 
        hvg_candidates = adata.uns['SEDR_hvg_genes'].tolist()
    elif 'highly_variable' in adata.var: 
        hvg_candidates = adata.var_names[adata.var['highly_variable']].tolist()

    if gene_type == "ALL":
        target_genes = adata.var_names.tolist()
        
    elif gene_type in ["HVGs", "LVGs"]:
        if not hvg_candidates:
            raise ValueError(
                f"Cannot determine {gene_type} genes. Please ensure 'SEDR_hvg_genes' is in adata.uns "
                f"or 'highly_variable' is in adata.var."
            )
            
        if gene_type == "HVGs":
            target_genes = [g for g in hvg_candidates if g in adata.var_names]
            if not target_genes:
                 raise ValueError("HVGs list is empty after intersection with adata.var_names.")
            
        elif gene_type == "LVGs":
            # LVGs = ALL - HVGs
            all_genes_set = set(adata.var_names.tolist())
            hvg_set = set(hvg_candidates)
            target_genes = list(all_genes_set - hvg_set)
            if not target_genes:
                 raise ValueError("LVGs list is empty. All genes might be marked as HVGs.")
    else:
        raise ValueError(f"Invalid gene_type: {gene_type}. Must be 'HVGs', 'LVGs', or 'ALL'.")

    valid_genes = [g for g in target_genes if g in adata.var_names]
    gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
    
    if not valid_genes:
        raise ValueError(f"Target gene list is empty for type {gene_type}.")

    print(f"--- Calculating CV on {len(valid_genes)} genes of type: {gene_type} ---")
   
    type_weights = adata.obs[label_key].value_counts(normalize=True)
    all_results = {}
    calculating_methods = []

    for method in methods:
        if method in adata.layers:
            X_subset = adata.layers[method][:, gene_indices]
        else:
            print(f"Warning: Method '{method}' not found in adata.layers. Skipping.")
            continue

        calculating_methods.append(method)
    
        if sp.issparse(X_subset): X_subset = X_subset.toarray()
            
        df = pd.DataFrame(X_subset, columns=valid_genes)
        df['Batch'] = adata.obs[batch_key].values
        df['Type'] = adata.obs[label_key].values
        
        centroids = df.groupby(['Batch', 'Type']).mean()
        
        centroids_std = centroids.groupby(level='Type').std()
        centroids_mean = centroids.groupby(level='Type').mean()
        
        epsilon = 1e-10
        cv_per_type = centroids_std / (np.abs(centroids_mean) + epsilon)
        
        current_weights = type_weights.reindex(cv_per_type.index).fillna(0)
        weighted_cv = cv_per_type.multiply(current_weights, axis=0).sum(axis=0)
        
        all_results[method] = weighted_cv

    if not calculating_methods:
        print("Error: No valid methods found in adata.layers for CV calculation.")
        return pd.DataFrame()

    print(f"Methods successfully calculated: {calculating_methods}")

    return pd.DataFrame(all_results)

def create_shuffled_batches(adata, n_batches=5):
    if 'cell_names' in adata.obsm:
        del adata.obsm['cell_names']

    indices = np.random.permutation(adata.n_obs)
    batch_indices_list = np.array_split(indices, n_batches)
    
    adatas_list = []
    for i, idx in enumerate(batch_indices_list):
        tmp_adata = adata[idx].copy()
        tmp_adata.obs['sim_batch'] = f'batch_{i}'
        tmp_adata.obs_names = [f"{name}_s{i}" for name in tmp_adata.obs_names]
        adatas_list.append(tmp_adata)
    return ad.concat(adatas_list, join='inner')


