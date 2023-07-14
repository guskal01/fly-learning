import torch

def state_dict_to_vec(state_dict):
    return torch.cat([torch.flatten(x) for x in state_dict.values()])
