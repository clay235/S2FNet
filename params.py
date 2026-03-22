import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_stats(model, dataset_name, K, length, windowSize, pool_size, top_k, output_units):
    original_device = next(model.parameters()).device
    model_cpu = model.to('cpu')
    model_cpu.eval()
    
    input_shape = (1, K, windowSize, windowSize)
    
    input_tensor = torch.randn(1, *input_shape)
    flops_thop, params_thop = profile(model_cpu, inputs=(input_tensor,), verbose=False)
    
    model.to(original_device)
    
    params_manual = count_parameters(model)

    print("=" * 60)    
    print(f"  Total Parameters:  {params_thop:,}")
    print(f"  Effective Parameters: {params_manual:,}")
    print(f"  FLOPs: {flops_thop / 1e6:.2f} MFLOPs")
    print("=" * 60)
    
    return params_manual, flops_thop
