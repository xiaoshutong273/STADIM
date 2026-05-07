import numpy as np
import itertools
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import hnswlib
import random

# ===================== Find Neighbors ======================
def knn(ds1, ds2, names1, names2, k=10, metric="cosine"):
    if metric == "cosine":
        nn_ = NearestNeighbors(n_neighbors=k, metric='cosine')
    else:
        nn_ = NearestNeighbors(n_neighbors=k, p=2, metric='minkowski')

    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        start_index = 1 if names2[b[0]] == names1[a] else 0
        for b_i in b[start_index:]:
            match.add((names1[a], names2[b_i]))
    return match

def nn_annoy(ds1, ds2, names1, names2, k=20, metric="cosine", n_trees=50, seed=2025):
    # Find nearest neighbors using Annoy
    # Build index.
    if metric=="cosine":
        tree = AnnoyIndex(ds2.shape[1], metric='angular') #angular means cosian
    else:
        tree = AnnoyIndex(ds2.shape[1], metric='euclidean')

    tree.set_seed(seed)
    for i in range(ds2.shape[0]):
        tree.add_item(i, ds2[i, :])
    tree.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(tree.get_nns_by_vector(ds1[i, :], k, search_k=-1))
    ind = np.array(ind)
    
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        start_index = 1 if names2[b[0]] == names1[a] else 0
        for b_i in b[start_index:]:
            match.add((names1[a], names2[b_i]))
    return match

def nn_hnsw(ds1, ds2, names1, names2, k=20, metric="cosine", ef_construction=200, M=16, seed=2025):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    
    if metric == "cosine":
        index = hnswlib.Index(space='cosine', dim=dim)
    else:
        index = hnswlib.Index(space='l2', dim=dim)

    index.set_num_threads(1)

    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M, random_seed=seed)
    index.add_items(ds2)
    index.set_ef(50)

    index.set_num_threads(1)
    
    ind, _ = index.knn_query(ds1, k=k)
    
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        start_index = 1 if names2[b[0]] == names1[a] else 0
        for b_i in b[start_index:]:
            match.add((names1[a], names2[b_i]))
    return match

def mnn(ds1, ds2, names1, names2, k=20, metric="cosine", approx = 'hnsw', n_trees=50, ef_construction=200, M=16, seed=2025):
    # Find nearest neighbors in first direction.
    if not approx:
        result1 = knn(ds1, ds2, names1, names2, k=k, metric=metric)
        result2 = knn(ds2, ds1, names2, names1, k=k, metric=metric)
    elif approx == "annoy":
        result1 = nn_annoy(ds1, ds2, names1, names2, k=k, metric=metric, n_trees=n_trees, seed=seed)
        result2 = nn_annoy(ds2, ds1, names2, names1, k=k, metric=metric, n_trees=n_trees, seed=seed)
    elif approx == "hnsw":
        result1 = nn_hnsw(ds1, ds2, names1, names2, k=k, metric=metric, ef_construction=ef_construction, M=M, seed=seed)
        result2 = nn_hnsw(ds2, ds1, names2, names1, k=k, metric=metric, ef_construction=ef_construction, M=M, seed=seed)
    else:
        raise ValueError("Invalid 'approx' argument. Must be False, 'annoy', or 'hnsw'.")
    
    # Compute mutual nearest neighbors.
    result2_swapped = set((b, a) for a, b in result2)
    intersection = result1.intersection(result2_swapped)
    # intersection = result1 & set([(b, a) for a, b in result2])  #intersect
    mutual = intersection | set((b, a) for a, b in intersection)
    return mutual # {('b1', 'a1'), ('a1', 'b1'), ('a2', 'b2'), ('b2', 'a2')}

# ===================== Find Positives ======================
def all_ap(adata, knn_c=6, knn_e=10, mnn_n=20, approx = 'hnsw', n_trees=50, ef_construction=200, M=16, seed=2025):
    knn_c = knn_c + 1
    knn_e = knn_e + 1
    mnn_n = mnn_n + 1
    slice_use = np.unique(adata.obs['sample'])
    emb = {}
    ccs = {}
    cells = {}
    for sample_t in slice_use:
        mask = (adata.obs['sample'] == sample_t).values 
        emb[sample_t] = adata.obsm['X_pca'][mask]
        ccs[sample_t] = adata.obsm['S_scale'][mask]
        cells[sample_t] = adata.obsm['cell_names'][mask] # obs_names[mask].values

    keys = list(emb.keys())
    keys.sort()
    matchs = set()

    for i in range(len(keys)):
        k1 = keys[i]
        if not approx:
            result1 = knn(ccs[k1], ccs[k1], cells[k1], cells[k1], k=knn_c, metric="ed")
            result2 = knn(emb[k1], emb[k1], cells[k1], cells[k1], k=knn_e, metric="cosine")
        elif approx == 'annoy':
            result1 = nn_annoy(ccs[k1], ccs[k1], cells[k1], cells[k1], k=knn_c, metric="ed", n_trees=n_trees, seed=seed)
            result2 = nn_annoy(emb[k1], emb[k1], cells[k1], cells[k1], k=knn_e, metric="cosine", n_trees=n_trees, seed=seed)
        else: # hnsw
            result1 = nn_hnsw(ccs[k1], ccs[k1], cells[k1], cells[k1], k=knn_c, metric="ed", ef_construction=ef_construction, M=M, seed=seed)
            result2 = nn_hnsw(emb[k1], emb[k1], cells[k1], cells[k1], k=knn_e, metric="cosine", ef_construction=ef_construction, M=M, seed=seed)
        matchs.update(result1.union(result2))

    for comb in itertools.combinations(range(len(keys)), 2):
        i, j = comb
        k1, k2 = keys[i], keys[j]
        mutual = mnn(emb[k1], emb[k2], cells[k1], cells[k2], k=mnn_n, metric="cosine", approx=approx, n_trees=n_trees, ef_construction=ef_construction, M=M, seed=seed)
        matchs.update(mutual)

    aps_dict = defaultdict(list)
    sorted_matchs = sorted(list(matchs))
    for i, j in sorted_matchs:
        aps_dict[i].append(j)
    aps_dict = dict(aps_dict)
   
    return aps_dict, cells
