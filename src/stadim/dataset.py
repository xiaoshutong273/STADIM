import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from scipy.sparse import issparse, csr_matrix, diags, vstack
import multiprocessing as mp
import time
import scanpy as sc
import random
from tqdm import tqdm
import anndata as ad
import math

# ===================== Prepare Dataset =====================
def pre_dataset(sdata, aps_dict, cells, layer='X_norm', seed=2025):

    random.seed(seed)
    np.random.seed(seed)
        
    is_sorted = sdata.obs_names.is_monotonic_increasing
    if not is_sorted:
        error_msg = (
            "CRITICAL ERROR: sdata.obs_names is not sorted. "
            "According to the required preprocessing pipeline, all cells must be "
            "sorted by obs_names before find neighbours. "
            "Please apply 'sdata = sdata[sdata.obs_names.argsort()].copy()' earlier in your script."
        )
        raise ValueError(error_msg)
    cells_for_train = sdata.obs_names.tolist()
    
    n_cells, n_vars = sdata.shape
    total_elements = n_cells * n_vars
    threshold = 1 * 10**10 

    if total_elements > threshold:
        print(f"    --- Data size ({total_elements:.2e}) > 1e10. Use X_norm as input data.")
        input_data = sdata.layers['X_norm'].copy()
    else:
        if layer not in sdata.layers: raise ValueError(f"Layer '{layer}' not found in sdata.layers.")
        input_data = sdata.layers[layer].copy()

    if issparse(input_data):
        if not isinstance(input_data, csr_matrix): input_data = input_data.tocsr()
        input_data.sort_indices()
    input_data = input_data.astype(np.float32)

    if 'size_factor' not in sdata.obs.columns: raise ValueError("sdata.obs lacks 'size_factor'.")
    diag_si = torch.from_numpy(sdata.obs['size_factor'].values).float()

    if 'sample' not in sdata.obs.columns: raise ValueError("sdata.obs lacks 'sample' which means batch label.")
    batch_codes = sdata.obs['sample'].astype('category').cat.codes.values
    batches = torch.from_numpy(batch_codes).long()
    sample_mapping = dict(enumerate(sdata.obs['sample'].astype('category').cat.categories))
    print(sample_mapping)
                            
    if 'X_raw' not in sdata.layers: raise ValueError("sdata.layers lacks 'X_raw'.")
    y_true = sdata.layers['X_raw']
    if issparse(y_true):
        if not isinstance(y_true, csr_matrix): y_true = y_true.tocsr()
        y_true.sort_indices()

    cell_to_idx = {name: idx for idx, name in enumerate(cells_for_train)}
    sample_to_indices = {} 
    for s_name, c_names in cells.items():
        sample_to_indices[s_name] = np.array([cell_to_idx[n] for n in c_names if n in cell_to_idx], dtype=np.int32)

    indexed_aps = {}
    for anc_name, pos_names in aps_dict.items():
        if anc_name in cell_to_idx:
            a_idx = cell_to_idx[anc_name]
            p_indices = [cell_to_idx[p] for p in pos_names if p in cell_to_idx]
            if p_indices:
                indexed_aps[a_idx] = np.array(p_indices, dtype=np.int32)

    all_sample_labels = sdata.obs['sample'].values
    
    anchor_to_positives = {}
    anchor_to_negatives = {}
    valid_anchors = []

    sorted_anchors = sorted(indexed_aps.keys())

    for anc_idx in tqdm(sorted_anchors, desc="Preparing triplet"):
        pos_indices = indexed_aps[anc_idx]
        current_sample = all_sample_labels[anc_idx]
        all_indices = sample_to_indices[current_sample]

        bad_set = set(pos_indices)
        bad_set.add(anc_idx)
        if len(all_indices) > 600:
            candidates = np.random.choice(all_indices, 600, replace=False)
            neg_indices = [x for x in candidates if x not in bad_set]
            if len(neg_indices) > 500:
                neg_indices = neg_indices[:500]
        else:
            neg_indices = [x for x in all_indices if x not in bad_set]
            if len(neg_indices) > 500:
                neg_indices = random.sample(neg_indices, 500)

        if neg_indices:
            anchor_to_positives[anc_idx] = pos_indices
            anchor_to_negatives[anc_idx] = np.array(neg_indices, dtype=np.int32)
            valid_anchors.append(anc_idx)

    return input_data, diag_si, batches, y_true, anchor_to_positives, anchor_to_negatives, valid_anchors

# ================== IterableTripletDataset ==================
class IterableTripletDataset(IterableDataset):
    def __init__(self, input_data, diag_si, batches, y_true, 
                 anchor_to_positives, anchor_to_negatives, valid_anchors,
                 update_ratio=0.8, batch_size=256, seed=2025):
        
        self.input_data = input_data 
        self.diag_si = np.array(diag_si, dtype=np.float32)
        self.batches = np.array(batches, dtype=np.int64)
        self.y_true = y_true 

        self.anchor_to_positives = anchor_to_positives
        self.anchor_to_negatives = anchor_to_negatives
        self.valid_anchors = np.array(valid_anchors)
        self.n_anchors = len(valid_anchors)
        self.update_ratio = float(update_ratio)  

        self.batch_size = batch_size
        self.seed = seed
        self._set_seed(self.seed)

        # Shared memory for triplets: [anchor_idx, pos_idx, neg_idx]
        self.shared_triplets = mp.Array('i', self.n_anchors * 3)
        self.anchor_map = {a: i for i, a in enumerate(valid_anchors)}  
        # Epoch counter for dynamic updates
        self.epoch_counter = mp.Value('i', 0)

        self._initialize_triplets()
        print(f"IterableTripletDataset Initialized: {self.n_anchors} anchors, batch_size={batch_size}")

    def _set_seed(self, seed):
        """Helper to set seed for both random and numpy"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _initialize_triplets(self):
        """Initialize triplets into shared memory array"""
        print("Initializing triplets...")
        for a_idx, arr_idx in self.anchor_map.items():
            pos = self.anchor_to_positives[a_idx]
            neg = self.anchor_to_negatives[a_idx]

            sub_seed = self.seed + arr_idx  # Deterministic initialization per anchor
            np.random.seed(sub_seed)
            
            base = arr_idx * 3
            self.shared_triplets[base] = a_idx
            self.shared_triplets[base + 1] = np.random.choice(pos)
            self.shared_triplets[base + 2] = np.random.choice(neg)

        self._set_seed(self.seed) # Reset to global seed

    def update_epoch_triplets(self):
        """
        Dynamically update a portion of positive/negative pairs for the next epoch.
        """
        with self.epoch_counter.get_lock():
            self.epoch_counter.value += 1
        current_epoch = self.epoch_counter.value

        n_update = max(1, int(float(self.n_anchors) * float(self.update_ratio)))
        np.random.seed(self.seed + current_epoch)
        update_indices = np.random.choice(self.n_anchors, n_update, replace=False)
        
        for arr_idx in update_indices:
            a_idx = self.valid_anchors[arr_idx]
            pos = self.anchor_to_positives[a_idx]
            neg = self.anchor_to_negatives[a_idx]

            sub_seed = self.seed + current_epoch + arr_idx
            np.random.seed(sub_seed)
            
            base = arr_idx * 3
            self.shared_triplets[base + 1] = np.random.choice(pos)
            self.shared_triplets[base + 2] = np.random.choice(neg)

        self._set_seed(self.seed) # Reset
        return n_update

    def __iter__(self):
        """Generator to yield batches"""
        worker_info = get_worker_info()
        local_triplets = np.frombuffer(self.shared_triplets.get_obj(), dtype=np.int32).reshape(-1, 3).copy()

        if worker_info is not None:
            # Multi-process: Split data among workers
            worker_seed = self.seed + (self.epoch_counter.value * 100) + worker_info.id
            self._set_seed(worker_seed)
            local_triplets = local_triplets[worker_info.id::worker_info.num_workers]
        else:
            # Single-process
            iter_seed = self.seed + (self.epoch_counter.value * 100)
            self._set_seed(iter_seed)
        
        np.random.shuffle(local_triplets)
        
        n_samples = len(local_triplets)
        is_input_sparse = issparse(self.input_data)
        is_y_sparse = issparse(self.y_true)
        
        for i in range(0, n_samples, self.batch_size):
            batch_triplets = local_triplets[i:i + self.batch_size]
            if len(batch_triplets) == 0: continue
            
            anc_idx = batch_triplets[:, 0]
            pos_idx = batch_triplets[:, 1]
            neg_idx = batch_triplets[:, 2]
            
            def to_tensor(data, idxs, is_sparse):
                if is_sparse:
                    return torch.from_numpy(data[idxs].toarray()).float()
                return torch.from_numpy(data[idxs]).float()

            yield (
                to_tensor(self.input_data, anc_idx, is_input_sparse),
                to_tensor(self.input_data, pos_idx, is_input_sparse),
                to_tensor(self.input_data, neg_idx, is_input_sparse),
                torch.from_numpy(self.diag_si[anc_idx]).float(),
                torch.from_numpy(self.batches[anc_idx]).long(),
                to_tensor(self.y_true, anc_idx, is_y_sparse)
            )

# ===================== Margin choose ========================
def calculate_recommended_margin(dataset, target_ratio=0.7, n_runs=5):
    """
    Calculate the recommended margin by averaging results from multiple runs.
    
    Args:
        dataset: IterableTripletDataset containing triplet data.
        target_ratio (float): Target ratio for hard triplets (default 0.7).
        n_runs (int): Number of times to calculate and average (default 5).
        
    Returns:
        float: The recommended margin value.
    """
    print(f"Calculating recommended margin (averaging over {n_runs} runs)...")
    margins = []
    initial_epoch = dataset.epoch_counter.value
    base_seed = dataset.seed

    threshold = 15.0

    for run in range(n_runs):
        # print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # 1. Update triplets to get a fresh random batch
        dataset.epoch_counter.value = initial_epoch + run + 1000
        dataset.update_epoch_triplets()
        run_seed = base_seed + 10000 + run
        np.random.seed(run_seed)

        # 2. Sample data (20k samples for speed)
        local_triplets = np.frombuffer(dataset.shared_triplets.get_obj(), dtype=np.int32).reshape(-1, 3)
        total_samples = len(local_triplets)
        
        if total_samples <= 20000:
            sampled_triplets = local_triplets
        else:
            sample_indices = np.random.choice(len(local_triplets), 20000, replace=False)
            sampled_triplets = local_triplets[sample_indices]

        # 3. Retrieve vectors
        if issparse(dataset.input_data):
            anchors = dataset.input_data[sampled_triplets[:, 0]].toarray()
            positives = dataset.input_data[sampled_triplets[:, 1]].toarray()
            negatives = dataset.input_data[sampled_triplets[:, 2]].toarray()
        else:
            anchors = dataset.input_data[sampled_triplets[:, 0]]
            positives = dataset.input_data[sampled_triplets[:, 1]]
            negatives = dataset.input_data[sampled_triplets[:, 2]]

        # 4. Calculate L2 distances
        pos_dist = np.linalg.norm(anchors - positives, ord=2, axis=1)
        neg_dist = np.linalg.norm(anchors - negatives, ord=2, axis=1)
        diffs = neg_dist - pos_dist

        # 5. Outlier Filtering (IQR Method)
        Q1, Q3 = np.percentile(diffs, [25, 75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        filtered_diffs = diffs[(diffs >= lower) & (diffs <= upper)]

        if len(filtered_diffs) == 0:
            margins.append(0.0)
            continue

        # 6. Calculate Stats (Separation & Margin)
        separation_ratio = np.mean(neg_dist > pos_dist)
        sorted_diffs = np.sort(filtered_diffs)
        # Select margin at the target quantile (e.g., 70th percentile of difficulty)
        current_margin = sorted_diffs[int(len(sorted_diffs) * target_ratio)]
        current_margin = max(0.0, current_margin)
        # print(f"  Separation Ratio (Neg > Pos): {separation_ratio:.2%}")
        # print(f"  Calculated Margin: {current_margin:.4f}")
        
        if current_margin <= threshold:
            margins.append(current_margin)
        else:
            adjusted_margin = threshold
            # print(f"  Adjusted Margin: {adjusted_margin:.4f}")
            margins.append(adjusted_margin)

    dataset.epoch_counter.value = initial_epoch
    # 7. Final Average
    avg_margin = np.mean(margins)
    print(f"\n{'='*40}")
    print(f"Margins from {n_runs} runs: {[round(m, 4) for m in margins]}")
    print(f"Final Recommended Margin: {avg_margin:.4f}")
    
    return avg_margin