# dataset, train and eval
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy
import copy
import yaml
from exp.result_io import save_acc, rep_save_model

# models
# from model.softmedian import SoftMedianAPPNP, SoftMedianPropagation
from model.mlp import MLP
from model.att_func import get_default_att_func, get_log_att_func, get_mask_att_func, get_step_p_norm_att_func, get_soft_step_l21_att_func, get_mcp_att_func, get_scad_att_func
from model.rung import RUNG

# preprocessing for sontructing models
# from model.plumbing import GraphSequential, PreprocessA, PreprocessX
from collections import OrderedDict

# computation pkgs
import torch
from torch import nn
import numpy as np

import re



def get_model_default_citeseer(
    model_name, custom_model_params={}, custom_fit_params={}, as_paper=True, seed=None, D=None, device=None
):
    if device is None:
        # prefer cuda when available
        # PyTorch 1.12.1 with CUDA 11.3 supports compute capability 5.0+
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                print(f"Detected GPU: {props.name} with compute capability {props.major}.{props.minor}")
                if props.major < 5:
                    print(f"GPU not supported; falling back to CPU.")
                    device = torch.device('cpu')
                else:
                    device = torch.device('cuda')
                    print(f"Using GPU: {props.name}")
            except Exception as e:
                print(f"Error querying GPU: {e}. Using CPU.")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    
    torch.manual_seed(0 if seed is None else seed)

    A, X, y = get_dataset("citeseer")
    sp = get_splits(y)
    train_idx, val_idx, test_idx = sp[0]

    D = X.shape[1] if D is None else D
    C = y.unique().shape[0]

    if model_name == 'RUNG':
        return RUNG(D, C, [64], get_mcp_att_func(custom_model_params['gamma']), 0.9).to(device), custom_fit_params
