import os
import torch
import numpy as np
from scipy.sparse import issparse, csr_matrix
import time

def run_inference_and_save(trained_model, input_data, batches, max_val, sdata, save_dir, device, 
                           save_vars_dict, encoder_layers, loss_mode='nb'):
    # print("\n=== 8. Generating denoised expression...")
    trained_model.to(device)
    trained_model.eval()

    # 1. Check data scale
    n_cells, n_genes = input_data.shape
    total_elements = n_cells * n_genes
    is_large_data = total_elements > 1 * 10**10
    n_batches = len(np.unique(batches))

    # 2. Pre-allocate memory
    embed_dim = encoder_layers[-1]
    embed_array = np.zeros((n_cells, embed_dim), dtype=np.float32)
    
    if is_large_data:
        print(f"   !!! Data size ({total_elements:.2e}) exceeds threshold (1 * 10**10).")
        print("   !!! Denoised matrix will NOT be computed or stored to avoid RAM OOM.")
        marginalized_mean = None
        output_withbatch = None
    else:
        print(f"   Data size ({total_elements:.2e}) is within limits. Allocating memory for denoised data.")
        marginalized_mean = np.zeros((n_cells, n_genes), dtype=np.float32)
        output_withbatch = np.zeros((n_cells, n_genes), dtype=np.float32)

    # 3. Batch Inference
    if issparse(input_data) and not isinstance(input_data, csr_matrix):
        input_data = input_data.tocsr()

    eval_batch_size = 2048
    with torch.no_grad():
        for start_idx in range(0, n_cells, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, n_cells)
            
            batch_raw = input_data[start_idx:end_idx]
            if issparse(batch_raw):
                batch_raw = batch_raw.toarray()
            
            input_tensor = torch.from_numpy(batch_raw).float().to(device)
            current_bs = input_tensor.size(0)
            true_batch_index = torch.as_tensor(batches[start_idx:end_idx], device=device).long()

            if loss_mode == 'nb':
                emb, output, _ = trained_model(input_tensor, batch_index=true_batch_index)#, mode='train')
            else: # zinb
                emb, output, _, _ = trained_model(input_tensor, batch_index=true_batch_index)#, mode='train')
            
            embed_array[start_idx:end_idx] = emb.cpu().numpy()

            if not is_large_data:
                # denoised_array[start_idx:end_idx] = mean.cpu().numpy()
                # denoised_array[start_idx:end_idx] = np.clip(mean.cpu().numpy(), a_min=None, a_max=max_val)
                output_withbatch[start_idx:end_idx] = np.clip(output.cpu().numpy(), a_min=None, a_max=max_val)

            if not is_large_data:
                chunk_sum = np.zeros((current_bs, n_genes), dtype=np.float32)
                
                for b_id in range(n_batches):
                    fake_batch_idx = torch.full((current_bs,), b_id, dtype=torch.long, device=device)
                    outputs = trained_model(input_tensor, batch_index=fake_batch_idx)#, mode='train')
                    marg_mean = outputs[1] 
                    chunk_sum += np.clip(marg_mean.cpu().numpy(), a_min=None, a_max=max_val)
                # Average across all possible batches
                marginalized_mean[start_idx:end_idx] = chunk_sum / n_batches
            
            del input_tensor, true_batch_index, emb, output
            if not is_large_data:
                del fake_batch_idx, outputs, marg_mean

    if not is_large_data and marginalized_mean is not None:
        sdata.layers['STADIM'] = marginalized_mean
        sdata.layers['STADIM_withbatch'] = output_withbatch
    sdata.obsm['STADIM'] = embed_array
        
    # print("\n=== 9. Save data...")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
        if not is_large_data and marginalized_mean is not None:
            np.save(os.path.join(save_dir, "STADIM_denoised.npy"), marginalized_mean)
        
        np.save(os.path.join(save_dir, "STADIM_embed.npy"), embed_array)
        
        sdata.X = sdata.layers['X_raw'].copy()
        sdata.write_h5ad(os.path.join(save_dir, "sdata.h5ad"))
        torch.save(trained_model.cpu().state_dict(), os.path.join(save_dir, "trained_model.pth"))
        
        # Save other variables
        clean_vars = {}
        for k, v in save_vars_dict.items():
            if isinstance(v, torch.Tensor):
                clean_vars[k] = v.cpu().numpy()
            elif issparse(v):
                print(f"   Warning: '{k}' is sparse. Saving sparse matrices inside npz is skipped to avoid errors.")
                clean_vars[k] = "Sparse Matrix (Skipped)" 
            else:
                clean_vars[k] = v
    
        np.savez(os.path.join(save_dir, "other_variables.npz"), **clean_vars)
        print(f"All results saved to {save_dir}")
        
    return sdata