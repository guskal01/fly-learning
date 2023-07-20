from constants import *
from models import Net
from utils import *
import torch.nn.functional as F

from torch.utils.data import DataLoader

class SimilarModel():
    def __init__(self, stealthiness, multiply_changes=SELECT_CLIENTS):
        self.stealthiness = stealthiness
        self.multiply_changes = multiply_changes

    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.train()
        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            batch_train_losses = []
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output = net(data)
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        good_net_params = net_to_params(net).detach().clone()

        bad_net = Net().to(device)
        bad_net.load_state_dict(net.state_dict())
        bad_opt = torch.optim.Adam(bad_net.parameters(), lr=1e-4)
        for epoch in range(1, 15+1):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                bad_opt.zero_grad()
                output = bad_net(data)
                loss = -net.loss_fn(output, target)
                #print(f"{loss.item():.4f} {self.stealthiness * F.mse_loss(net_to_params(bad_net), good_net_params).item():.4f}")
                loss += self.stealthiness * F.mse_loss(net_to_params(bad_net), good_net_params)
                loss.backward()
                bad_opt.step()
        
        sd = bad_net.state_dict()
        for key in sd:
            sd[key] = net.state_dict()[key] + (sd[key]-net.state_dict()[key]) * self.multiply_changes
        net.load_state_dict(sd)

        return epoch_train_losses
