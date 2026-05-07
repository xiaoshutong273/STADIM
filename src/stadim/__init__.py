"""
STADIM: Adaptive Deep Metric Learning for Denoising and Integration of Multi-Slice Spatial Transcriptomics
A Python package for spatial transcriptomics data denoising
"""

__version__ = "0.1.0"
__author__ = "shutong xiao"

from .preprocess import read_data, MY_preprocess
from .triplets import all_ap
from .dataset import pre_dataset, IterableTripletDataset, calculate_recommended_margin
from .network import STADIM
from .trainer import train
from .save import run_inference_and_save
from .evals import calculate_tau_index, calculate_fold_change, calculate_cohens_d, calculate_new_morani, calculate_sns_score, calculate_batch_entropy, calculate_cv, create_shuffled_batches
from .run_STADIM import run_STADIM

__all__ = [
    'read_data',
    'MY_preprocess',
    'all_ap',
    'pre_dataset',
    'IterableTripletDataset',
    'calculate_recommended_margin',
    'STADIM',
    'train',
    'run_inference_and_save',
    'calculate_tau_index', 
    'calculate_fold_change',
    'calculate_cohens_d',
    'calculate_new_morani',
    'calculate_sns_score',
    'calculate_batch_entropy',
    'calculate_cv',
    'create_shuffled_batches',
    'run_STADIM'
]