import warnings
warnings.filterwarnings("ignore")
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from typing import Union, Optional, List
from sklearn.preprocessing import MinMaxScaler
import gc
from scipy.sparse import issparse, vstack

# =============== Read Data and Filter Gene =================
def read_data(
    data_input: Union[ad.AnnData, List],
    sample_names: Optional[Union[str, List[str]]] = None,
    batch_key: Optional[str] = None, 
    min_genes: int = 0,
    min_cells: int = 10,  
) -> ad.AnnData:
    # ========== Handle Single/Multi-slice Input ==========
    if not isinstance(data_input, list):
        input_list = [data_input]
    else:
        input_list = data_input
    num_samples = len(input_list)
    is_multi = num_samples > 1
    
    adata_list = []
    spatial_dict = {}
    
    if is_multi:
        print(f"Detected multi-slice data, total {num_samples} slices")
        if sample_names is None:
            final_sample_names = [f"S{i+1}" for i in range(num_samples)]
        elif not isinstance(sample_names, list):
            final_sample_names = [sample_names] * num_samples
        else:
            final_sample_names = sample_names

        if len(final_sample_names) != num_samples:
            raise ValueError("Number of sample names does not match number of slices")
        
        for i, (data, name) in enumerate(zip(input_list, final_sample_names)):
            if isinstance(data, str):
                adata = ad.read_h5ad(data)
            else:
                adata = data.copy()
            
            if batch_key is not None and batch_key in adata.obs.columns:
                print(f"  Slice {i+1}: Using user-defined column '{batch_key}' as sample")
                if 'sample' in adata.obs.columns and batch_key != 'sample':
                    adata.obs['sample_original'] = adata.obs['sample']
                adata.obs['sample'] = adata.obs[batch_key]
                existing_name = adata.obs['sample'].unique()[0]
                print(f"  Slice {i+1}: Keeping existing sample label {existing_name}")
            else:
                adata.obs['sample'] = name
                print(f"  Slice {i+1}: Adding sample label {name}")

            # Handle unique obs names to prevent duplicates during merge
            sid = str(adata.obs['sample'].unique()[0])
            adata.obs_names = sid + "_" + adata.obs_names.astype(str)
            
            adata_list.append(adata)

            # Handle spatial info
            if 'spatial' in adata.uns and isinstance(adata.uns['spatial'], dict):
                if sid in adata.uns['spatial']:
                    spatial_dict[sid] = adata.uns['spatial'][sid]
                elif len(adata.uns['spatial']) == 1:
                    default_key = list(adata.uns['spatial'].keys())[0]
                    spatial_dict[sid] = adata.uns['spatial'][default_key]
                else:
                    spatial_dict[sid] = adata.uns['spatial']
            else:
                print(f"  Warning: Slice {sid} has no valid uns['spatial'] info")
    
    else:
        single_data = input_list[0]

        if isinstance(single_data, str):
            adata = ad.read_h5ad(single_data)
        else:
            adata = single_data.copy()
        
        if batch_key is not None and batch_key in adata.obs.columns:
            print(f"Processing single slice: Using user-defined column '{batch_key}'")
            if 'sample' in adata.obs.columns and batch_key != 'sample':
                adata.obs['sample_original'] = adata.obs['sample']
            adata.obs['sample'] = adata.obs[batch_key]
            existing_name = adata.obs['sample'].unique()[0]
            print(f"Processing single slice: Keeping existing sample label {existing_name}")
        else:
            if sample_names:
                sample_name = sample_names[0] if isinstance(sample_names, list) else sample_names
            else:
                sample_name = "S1"
            
            adata.obs['sample'] = sample_name
            print(f"Processing single slice: Adding sample label {sample_name}")
        
        # Unique obs names
        sid = str(adata.obs['sample'].unique()[0])
        adata.obs_names = sid + "_" + adata.obs_names.astype(str)

        adata_list = [adata]

        if 'spatial' in adata.uns and isinstance(adata.uns['spatial'], dict) and len(adata.uns['spatial']) > 0:
            if sid in adata.uns['spatial']:
                spatial_dict[sid] = adata.uns['spatial'][sid]
            else:
                spatial_dict[sid] = next(iter(adata.uns['spatial'].values()))
    
    # 2. Check Spatial Info
    # print("\nChecking Spatial Info:")
    # for sid in spatial_dict.keys():
    #     print(f"  Sample {sid}: keys -> {list(spatial_dict[sid].keys())}")

    # 3. Merging data and filter
    print("\nMerging data...")
    if len(adata_list) > 1:
        adata = ad.concat(adata_list, axis=0, join='outer', merge='same') 
        del adata_list
        gc.collect()
        
        if spatial_dict:
            adata.uns['spatial'] = spatial_dict
            # print(f"  Merged spatial info: {list(spatial_dict.keys())}")
    else:
        adata = adata_list[0]
        if spatial_dict:
            adata.uns['spatial'] = spatial_dict
            # print(f"  Merged spatial info: {list(spatial_dict.keys())}")

    # Fill NAs with 0 if outer join created NaNs
    if issparse(adata.X): 
        if np.isnan(adata.X.data).any():
            adata.X.data = np.nan_to_num(adata.X.data)
    else: 
        if np.isnan(adata.X).any(): 
            adata.X = np.nan_to_num(adata.X)

    print(f"\nRaw Merged Data: {adata.shape[0]} spots × {adata.shape[1]} genes")

    # Unified Filtering
    print(f"Unified Filtering (min_genes={min_genes}, min_cells={min_cells})...")
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = adata[np.argsort(adata.obs_names)]
    if min_genes > 0: sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells > 0: sc.pp.filter_genes(adata, min_cells=min_cells)

    if 'array_row' in adata.obs.columns:
        adata.obs['array_row'] = adata.obs['array_row'].astype(int)
    if 'array_col' in adata.obs.columns:
        adata.obs['array_col'] = adata.obs['array_col'].astype(int)

    
    p999 = np.percentile(adata.X.data if hasattr(adata.X, 'data') else adata.X[adata.X > 0], 99.99)
    if issparse(adata.X): 
        # print(f"  → X_raw max value : {adata.X.data.max()}")
        adata.X.data = np.clip(adata.X.data, 0, p999)
        # print(f"  → X_raw max value after clip 99.99%: {adata.X.data.max()}")
    else:
        # print(f"  → X_raw max value : {adata.X.max()}")
        adata.X = np.clip(adata.X, 0, p999)
        # print(f"  → X_raw max value after clip 99.99%: {adata.X.max()}")
        
    adata.layers['X_raw'] = adata.X.copy()
    
    print(f"  → After Filtering: {adata.shape[0]} spots × {adata.shape[1]} genes")
    print(f"  ✓ Complete: Samples included: {adata.obs['sample'].unique().tolist()}")
    print(f"  ✓ Complete: {adata}")
    return adata

# ===================== Data Preporcess =====================
def MY_preprocess(adata, nhvgs=2000, dim=50):
    # print("== Preprocess...")
    adata.X = adata.layers['X_raw'].copy()
    
    sc.pp.normalize_total(adata, target_sum=None)
    adata.layers['X_norm'] = adata.X.copy()
    
    sc.pp.log1p(adata)
    adata.layers['X_log'] = adata.X.copy()
    
    sc.pp.calculate_qc_metrics(adata, inplace=True, layer='X_raw', percent_top=None)
    adata.obs['size_factor'] = adata.obs.total_counts/np.median(adata.obs.total_counts)
    
    slice_use = np.unique(adata.obs['sample'])
    adata = adata[np.argsort(adata.obs_names)]

    # ============================================
    # print("== Selecting HVGs...")
    from collections import Counter
    hvg_counter = Counter()
    hvg_scores = {}
    
    for batch in slice_use:
        try:
            batch_data = adata[adata.obs['sample'] == batch].copy()
            sc.pp.highly_variable_genes(batch_data, flavor="seurat_v3", n_top_genes=min(batch_data.n_vars, nhvgs), layer="X_raw")
            
            hv_genes = batch_data.var_names[batch_data.var['highly_variable']].tolist()
            hvg_counter.update(hv_genes)
            
            for gene in hv_genes:
                rank_score = batch_data.var.loc[gene, 'highly_variable_rank']
                if gene not in hvg_scores:
                    hvg_scores[gene] = []
                hvg_scores[gene].append(rank_score)
    
            del batch_data
            gc.collect()
        except Exception as e:
            print(f"   Warning: HVG calculation failed for batch {batch}. Error: {e}")
            continue
    
    avg_ranks = {gene: np.mean(hvg_scores[gene]) for gene in hvg_counter}
    top_hvgs = sorted(hvg_counter.keys(), key=lambda g: (-hvg_counter[g], avg_ranks[g]))[:min(adata.n_vars, nhvgs)]
    adata.uns['top_hvgs'] = top_hvgs
    adata.var['highly_variable'] = adata.var_names.isin(top_hvgs)
    
    print(f"== Selected {len(top_hvgs)} HVGs across {len(slice_use)} slices")
    # ============================================

    # print("== Processing Spatial Scalings...")
    hvg_mask = adata.var['highly_variable']
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    batches = []
    for batch in slice_use:
        mask = adata.obs['sample'] == batch
        b_data = adata[mask, hvg_mask].copy()

        spatial_data = b_data.obsm['spatial']
        spatial_centered = spatial_data - np.mean(spatial_data, axis=0)
        b_data.obsm['S_scale'] = scaler.fit_transform(spatial_centered).astype(np.float32)
        sc.pp.scale(b_data, max_value=10, zero_center=True) # for PCA
        batches.append(b_data)
    adata_hvg_scaled = ad.concat(batches, merge="same")

    if not np.array_equal(adata_hvg_scaled.obs_names, adata.obs_names):
        print("   Resorting cells to match original order...")
        adata_hvg_scaled = adata_hvg_scaled[adata.obs_names].copy()
    
    adata.obsm['S_scale'] = adata_hvg_scaled.obsm['S_scale']
    adata.obsm['X_hvg_scale'] = adata_hvg_scaled.X.copy()
    # adata.uns['hvg_names'] = adata_hvg_scaled.var_names.tolist()

    # print("== Computing PCA...")
    pca_adata = ad.AnnData(adata_hvg_scaled.X)
    sc.tl.pca(pca_adata, n_comps=dim, random_state=2025) 
    adata.obsm['X_pca'] = pca_adata.obsm['X_pca'].copy()
    del adata_hvg_scaled, batches, pca_adata
    gc.collect()

    adata.obsm['cell_names'] = np.array(adata.obs_names)#.reshape(-1, 1)
    adata.X = adata.layers['X_norm'].copy()
    return adata