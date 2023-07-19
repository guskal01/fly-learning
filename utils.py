import torch
from models import Net
import copy

def state_dict_to_vec(state_dict):
    return torch.cat([torch.flatten(x) for x in state_dict.values()])

def vec_to_state_dict(vec, state_dict):
    i = 0
    for key in state_dict:
        pvec = vec[i : i+state_dict[key].nelement()]
        state_dict[key] = pvec.reshape(state_dict[key].shape)
        i += state_dict[key].nelement()

def test_conversions(state_dict):
    vec = state_dict_to_vec(state_dict)
    new_state_dict = copy.deepcopy(state_dict)
    for x in new_state_dict:
        new_state_dict[x].zero_()
        assert new_state_dict[x].flatten()[0] == 0
    vec_to_state_dict(vec, new_state_dict)
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
