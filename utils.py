import torch
from models import Net
import copy

def state_dict_to_vec(state_dict):
    for key in state_dict:
        if 'num_batches_tracked' in key:
            nbt = state_dict[key]
            break
    return torch.cat([torch.flatten(x) for key,x in state_dict.items() if 'num_batches_tracked' not in key]), nbt

def vec_to_state_dict(vec, state_dict, nbt):
    i = 0
    for key in state_dict:
        if 'num_batches_tracked' in key:
            state_dict[key] = nbt
            continue
        pvec = vec[i : i+state_dict[key].nelement()]
        state_dict[key] = pvec.reshape(state_dict[key].shape)
        i += state_dict[key].nelement()

def net_to_vec(net):
    return state_dict_to_vec(net.state_dict())

def vec_to_net(vec, nbt):
    net = Net()
    state_dict = net.state_dict()
    vec_to_state_dict(vec, state_dict, nbt)
    net.load_state_dict(state_dict)
    return net

def net_to_params(net):
    return torch.cat([torch.flatten(x) for x in net.parameters()])

def weighted_avg(net, client_nets, weights):
    state_dict = net.state_dict()
        
    for key in state_dict:
        state_dict[key] = sum([x[key]*weights[i] for i, x in enumerate(client_nets)])
    
    net.load_state_dict(state_dict)
    return net, None

