import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from .losses import NBLoss, ZINBLoss, TripletLoss
import random
import numpy as np
from torch.backends import cudnn
import functools

def fix_seed(seed=2025):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def global_worker_init_fn(worker_id=1, seed=2025):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
# ======================== train =============================
def train(dataset, model, margin=5, save_dir=None, device='cpu', seed=2025,
          epochs=100, lr=1e-3, loss_mode='nb', alpha_rec=1, beta_trip=0.1):

    fix_seed(seed)
    
    num_workers = min(mp.cpu_count() // 2, 8)

    worker_init_fn = functools.partial(global_worker_init_fn, seed=seed)
        
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers, 
                            prefetch_factor=num_workers//2 if num_workers > 0 else None,
                            persistent_workers=(num_workers > 0),
                            pin_memory=(torch.cuda.is_available() and device != 'cpu'),
                            worker_init_fn=worker_init_fn)

    model = model.to(device)
    recon_loss_fn = ZINBLoss().to(device) if loss_mode == 'zinb' else NBLoss().to(device)
    triplet_loss_fn = TripletLoss(margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Begin training: device={device}")
    history = {'recon': [], 'triplet': [], 'total': []} 
    model.train()
    
    with tqdm(total=epochs, desc="Training",ncols=100) as pbar:
        for epoch in range(epochs):
            if epoch > 0:
                dataset.update_epoch_triplets()
            
            epoch_recon, epoch_triplet = 0.0, 0.0
            batch_count = 0
            
            for batch in dataloader:
                anc, pos, neg, anc_sf, anc_batch, anc_true = [x.to(device, non_blocking=True) for x in batch]
                optimizer.zero_grad(set_to_none=True)
                
                if loss_mode == 'nb':
                    anc_z, anc_mean, anc_disp = model(anc, batch_index=anc_batch)#, mode='train')
                    recon_loss = recon_loss_fn(anc_true, anc_disp, anc_mean, anc_sf)
                else:
                    anc_z, anc_mean, anc_disp, anc_pi = model(anc, batch_index=anc_batch)#, mode='train')
                    recon_loss = recon_loss_fn(anc_true, anc_pi, anc_disp, anc_mean, anc_sf)

                pos_z = model(pos) # batch_index=None only return z 
                neg_z = model(neg)
                triplet_loss = triplet_loss_fn(anc_z, pos_z, neg_z)

                total_loss = alpha_rec * recon_loss + beta_trip * triplet_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                epoch_recon += recon_loss.item()
                epoch_triplet += triplet_loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_recon = epoch_recon / batch_count
                avg_triplet = epoch_triplet / batch_count
            else:
                avg_recon, avg_triplet = 0.0, 0.0
                
            history['recon'].append(avg_recon)
            history['triplet'].append(avg_triplet)   
            history['total'].append(alpha_rec * avg_recon + beta_trip * avg_triplet)      
            pbar.set_postfix({
                'recon': f'{avg_recon:.3f}',
                'triplet': f'{avg_triplet:.3f}',
                'total': f'{history["total"][-1]:.3f}'
            })
            pbar.update(1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(4, 3))
        for k, v in history.items():
            plt.plot(v, label=k)
        plt.legend()
        plt.grid()
        plt.title('Training Loss')
        plt.show()
        plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return model, history
