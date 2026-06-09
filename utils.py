import torch
from contextlib import nullcontext

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.999)) -> torch.optim.AdamW:
    device = next(model.parameters()).device
    use_fused = (device.type == 'cuda')

    decay, no_decay = [], []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=use_fused)
    return optimizer

def autocast_ctx(device):
    if device == 'cpu':
        return nullcontext()
    
    return torch.amp.autocast(device, dtype=torch.bfloat16)