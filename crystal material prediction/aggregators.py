import torch
from torch import Tensor
from torch_scatter import scatter
from typing import Optional

# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py

def aggregate_sum(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='min')

def aggregate_min_magnitude(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(torch.abs(src), index, 0, None, dim_size, reduce='min')

def aggregate_max(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='max')

def aggregate_max_magnitude(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(torch.abs(src), index, 0, None, dim_size, reduce='max')

def aggregate_product(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(torch.abs(src), index, 0, None, dim_size, reduce='mul')

def aggregate_var(src, index, dim_size):
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean

def aggregate_std(src, index, dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)

def aggregate_harmonic_mean(src, index, dim_size):
    return  torch.reciprocal(scatter(torch.reciprocal(src+ 1e-5), index, 0, None, dim_size, reduce='mean')+ 1e-5)

# def aggregate_geometric_mean(src, index, dim_size):
#     return  torch.exp(scatter(torch.abs(src), index, 0, None, dim_size, reduce='mul'))

def aggregate_Root_mean_square(src, index, dim_size):
    return  torch.sqrt(scatter(torch.pow(src,2)), index, 0, None, dim_size, reduce='mean')

def aggregate_Euclidean_norm(src, index, dim_size):
    return  torch.sqrt(scatter(torch.pow(src,2)), index, 0, None, dim_size, reduce='sum')

def aggregate_log_sum_exp(src, index, dim_size):
    return  torch.log(scatter(torch.exp(src), index, 0, None, dim_size, reduce='sum'))

AGGREGATORS = {
    'sum': aggregate_sum,
    'mean': aggregate_mean,
    'min': aggregate_min,
    'min_magnitude':aggregate_min_magnitude,
    'max': aggregate_max,
    'max_magnitude':aggregate_max_magnitude,
    'product':aggregate_product,
    'var': aggregate_var,
    'std': aggregate_std,
    'harmonic_mean':aggregate_harmonic_mean,
    #'geometric_mean':aggregate_geometric_mean,
    'Root_mean_square':aggregate_Root_mean_square,
    'Euclidean_norm':aggregate_Euclidean_norm,
    'log_sum_exp':aggregate_log_sum_exp
}