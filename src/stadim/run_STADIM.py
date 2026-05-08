import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import time
import multiprocessing as mp
import subprocess
import psutil
import pandas as pd
from datetime import datetime
import numpy as np

from .preprocess import *
from .triplets import *
from .dataset import *
from .network import *
from .trainer import *
from .save import *

class ResourceMonitor(mp.Process):
    def __init__(self, pid, device_id=None, log_path="resource_trace.txt", interval=1.0):
        super().__init__()
        self.target_pid = pid 
        self.interval = interval
        self.device_id = device_id
        self.log_path = log_path
        self.max_rss = mp.Value('f', 0.0) 
        self.max_vram = mp.Value('f', 0.0)
        self.stop_event = mp.Event()
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write("Time(s)\tMem_RSS(MB)\tGPU_VRAM(MB)\tCPU(%)\n")

    def get_full_memory(self):
        try:
            main_proc = psutil.Process(self.target_pid)
            mem = main_proc.memory_info().rss
            for child in main_proc.children(recursive=True):
                mem += child.memory_info().rss
            return mem / (1024 ** 2)
        except: return 0

    def get_gpu_memory_smi(self):
        if self.device_id is None: return 0
        try:
            cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {self.device_id}"
            return float(subprocess.check_output(cmd, shell=True).decode().strip())
        except: return 0

    def run(self):
        start_time = time.time()
        baseline_vram = self.get_gpu_memory_smi()
        while not self.stop_event.is_set():
            curr_time = round(time.time() - start_time, 1)
            rss = self.get_full_memory()
            vram = max(0, self.get_gpu_memory_smi() - baseline_vram)
            cpu = psutil.cpu_percent(interval=None)
            
            if rss > self.max_rss.value: self.max_rss.value = rss
            if vram > self.max_vram.value: self.max_vram.value = vram
            
            with open(self.log_path, 'a') as f:
                f.write(f"{curr_time}\t{rss:.2f}\t{vram:.2f}\t{cpu:.1f}\n")
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join()
        return {"max_rss_mb": round(self.max_rss.value, 2), "max_vram_mb": round(self.max_vram.value, 2)}

def run_STADIM(file_list, save_dir=None, sample_names=None, batch_key='sample', min_genes=0, min_cells=10, 
               device='cuda:0', seed=2026, nhvgs=2000, dim=50, save_preprocessed_h5ad=None, 
               knn_c=6, knn_e=10, mnn_n=25, layer='X_norm', triplets_update_ratio=0.8, batch_size=256, hard_triplets_ratio=0.7,
               epochs=100, lr=1e-3, loss_mode='nb', beta_trip=0.1, encoder_layers=[512, 256, 64], decoder_layers=[1000]):
    
    t0 = time.time()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")
    else:
        print(f"Results will be stored in adata.layers['STADIM']")

    print(f"\n=== 1. Begin read_data!")
    sdata = read_data(file_list, sample_names=sample_names, batch_key=batch_key, min_genes=min_genes, min_cells=min_cells)

    print(f"\n=== 2. Begin MY_preprocess!")
    sdata = MY_preprocess(sdata, nhvgs=nhvgs, dim=dim)

    if save_preprocessed_h5ad is not None:
        save_preprocess_h5ad_dir = os.path.dirname(save_preprocessed_h5ad)
        if save_preprocess_h5ad_dir:
            os.makedirs(save_preprocess_h5ad_dir, exist_ok=True)
        sdata.write_h5ad(save_preprocessed_h5ad)

    print(f"\n=== 3. Begin all_ap find neighbors!")
    aps_dict, cells = all_ap(sdata, knn_c=knn_c, knn_e=knn_e, mnn_n=mnn_n, approx='hnsw', n_trees=50, ef_construction=200, M=16, seed=seed)

    print(f"\n=== 4. Begin pre_dataset!")
    input_data, diag_si, batches, y_true, anchor_to_positives, anchor_to_negatives, valid_anchors = pre_dataset(
        sdata, aps_dict, cells, layer=layer, seed=seed)

    print(f"\n=== 5. Begin IterableTripletDataset!")
    dataset = IterableTripletDataset(input_data, diag_si, batches, y_true, 
                                     anchor_to_positives, anchor_to_negatives, valid_anchors,
                                     update_ratio=triplets_update_ratio, batch_size=batch_size, seed=seed)

    print(f"\n=== 6. Begin calculate_recommended_margin!")
    recommended_margin = calculate_recommended_margin(dataset, target_ratio=hard_triplets_ratio, n_runs=5)

    print(f"\n=== 7. Starting training...")
    input_dim = input_data.shape[1]
    n_batches = len(np.unique(batches))
    model = STADIM(input_dim=input_dim, n_batches=n_batches, encoder_layers=encoder_layers, decoder_layers=decoder_layers, distribution=loss_mode, seed=seed)
    max_val = float(sdata.layers['X_norm'].max())
    # print(f"    --- X_norm max value {max_val}")
    trained_model, history = train(dataset, model, recommended_margin, save_dir=save_dir, device=device, seed=seed,
                                   epochs=epochs, lr=lr, loss_mode=loss_mode, alpha_rec=1, beta_trip=beta_trip)

    print(f"\n=== 8. Generating denoised expression...")
    save_vars = {
        "aps_dict": aps_dict,
        "cells": cells,
        "anchor_to_positives": anchor_to_positives,
        "anchor_to_negatives": anchor_to_negatives,
        "valid_anchors": valid_anchors,
        "recommended_margin": recommended_margin,
        "loss_history": history,
        "train_params": {"data_dir": file_list, "save_dir": save_dir, "device": device, "seed": seed, 
                         "min_genes": min_genes, "min_cells": min_cells, "batch_key": batch_key,
                         "knn_c": knn_c, "knn_e": knn_e, "mnn_n": mnn_n, "input_layer": layer, "batch_size": batch_size, 
                         "triplets_update_ratio": triplets_update_ratio, "hard_triplets_ratio": hard_triplets_ratio,
                         "epochs": epochs, "lr": lr, "loss_mode": loss_mode, "beta_trip": beta_trip, 
                         "encoder_layers": encoder_layers, "decoder_layers": decoder_layers}
    }
    sdata = run_inference_and_save(
        trained_model=trained_model, input_data=input_data, batches=batches, max_val=max_val, 
        sdata=sdata, save_dir=save_dir, device=device, save_vars_dict=save_vars,
        encoder_layers=encoder_layers, loss_mode=loss_mode)

    print(f"\nAll processes finished! Total time: {(time.time() - t0)/60:.2f} mins.")
    return sdata

def main():
    parser = argparse.ArgumentParser(description="Run STADIM ...")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input h5ad file path(s)")
    parser.add_argument("--save_preprocessed_h5ad", type=str, default=None, help="Save pre.h5ad file path")
    parser.add_argument("--save_dir", type=str, default=None, help="Results directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cuda:0)")
    parser.add_argument("--monitor", action="store_true", help="Enable resource monitor")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    
    parser.add_argument("--sample_names", type=str, nargs='+', default=None, help="Names for input samples")
    parser.add_argument("--batch_key", type=str, default="sample", help="Key in obs for batch/sample information")
    parser.add_argument("--min_genes", type=int, default=0, help="Minimum genes per cell")
    parser.add_argument("--min_cells", type=int, default=10, help="Minimum cells per gene")
    parser.add_argument("--nhvgs", type=int, default=2000, help="Number of HVGs")
    parser.add_argument("--dim", type=int, default=50, help="PCA dimension")
    
    parser.add_argument("--knn_c", type=int, default=6, help="KNN for cells")
    parser.add_argument("--knn_e", type=int, default=10, help="KNN for edges/expansion")
    parser.add_argument("--mnn_n", type=int, default=25, help="MNN neighbors")
    parser.add_argument("--layer", type=str, default="X_norm", help="Layer to use for training (e.g. X_norm)")
    parser.add_argument("--hard_triplets_ratio", type=float, default=0.7, help="Hard triplets ratio")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--triplets_update_ratio", type=float, default=0.8, help="Triplets update ratio")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss_mode", type=str, default="nb", help="Loss mode (nb/zinb)")
    parser.add_argument("--beta_trip", type=float, default=0.1, help="Beta for triplet loss")
    parser.add_argument("--encoder_layers", type=int, nargs='+', default=[512,256,64], help="Encoder layers")
    parser.add_argument("--decoder_layers", type=int, nargs='+', default=[1000], help="Decoder layers")

    args = parser.parse_args()

    save_dir = args.save_dir
    device = args.device
    monitor_on = args.monitor

    input_basenames = [os.path.basename(f) for f in args.input]
    print(f"--- Processing {', '.join(input_basenames)} on {device} ---")
    print("\n" + "="*40)
    print("      [Hyperparameters Configuration]      ")
    print("="*40)
    for arg, value in vars(args).items():
        print(f"{arg:<25}: {value}")
    print("="*40 + "\n")

    monitor = None
    if monitor_on and save_dir is not None:
        gpu_id = int(device.split(':')[-1]) if 'cuda' in device else None
        trace_path = os.path.join(save_dir, f"resource_trace_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
        monitor = ResourceMonitor(os.getpid(), device_id=gpu_id, log_path=trace_path)
        monitor.start()
        print(f"Monitor started. Log: {trace_path}")

    start_t = time.time()
    success = False
    try:
        run_STADIM(file_list=args.input, save_dir=save_dir, sample_names=args.sample_names, batch_key=args.batch_key, 
                   min_genes=args.min_genes, min_cells=args.min_cells, device=device, seed=args.seed, nhvgs=args.nhvgs, dim=args.dim,
                   save_preprocessed_h5ad=args.save_preprocessed_h5ad, knn_c=args.knn_c, knn_e=args.knn_e, mnn_n=args.mnn_n,
                   layer=args.layer, triplets_update_ratio=args.triplets_update_ratio, batch_size=args.batch_size, 
                   hard_triplets_ratio=args.hard_triplets_ratio, epochs=args.epochs, lr=args.lr, loss_mode=args.loss_mode, 
                   beta_trip=args.beta_trip, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers)
        success = True
    except Exception as e:
        print(f"Error in execution: {e}")
        import traceback
        traceback.print_exc()

    elapsed = round((time.time() - start_t) / 60, 2)

    stats = {"max_rss_mb": 0, "max_vram_mb": 0}
    if monitor:
        stats = monitor.stop()
    
        print(f"Done. Time {elapsed}min, Mem {stats['max_rss_mb']}MB, GPU {stats['max_vram_mb']}MB")
        input_basenames = [os.path.basename(f) for f in args.input]
        perf_data = [{
            "sample": str(input_basenames), 
            "time_min": elapsed, 
            "mem_peak_mb": stats['max_rss_mb'], 
            "gpu_peak_mb": stats['max_vram_mb'],
            "status": "Success" if success else "Failed"
        }]
        pd.DataFrame(perf_data).to_csv(os.path.join(save_dir, "performance.csv"), index=False)


if __name__ == "__main__":
    main()
