import torch


def masked_mean_pool(inputs: torch.Tensor, mask: torch.Tensor, axis: int = 1):
    if len(mask.shape) != len(inputs.shape):
        mask = mask.unsqueeze(-1)
    mask = mask.to(inputs.device)
    return (inputs * mask).sum(axis=1) / mask.sum(axis=1)


def masked_max_pool(inputs: torch.Tensor, mask: torch.Tensor, axis: int = 1):
    if len(mask.shape) != len(inputs.shape):
        mask = mask.unsqueeze(-1)
    mask = mask.to(inputs.device)
    return (inputs * mask).max(axis=1)
