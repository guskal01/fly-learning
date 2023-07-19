import torch
from models import Net

def state_dict_to_vec(state_dict):
    return torch.cat([torch.flatten(x) for x in state_dict.values()])

def vec_to_state_dict(vec, state_dict):
    i = 0
    for key in state_dict:
        state_dict[key] = vec[i : i+state_dict[key].nelement()].reshape(state_dict[key].shape)
        i += state_dict[key].nelement()

def test_conversions(state_dict):
    vec = state_dict_to_vec(state_dict)
    new_state_dict = vec_to_state_dict(vec)
    for key in state_dict:
        assert torch.equal(state_dict[key], new_state_dict[key])


def net_to_vec(net):
    return state_dict_to_vec(net.state_dict())

def vec_to_net(vec):
    net = Net()
    state_dict = net.state_dict()
    vec_to_state_dict(vec, state_dict)
    net.load_state_dict(state_dict)
    return net
