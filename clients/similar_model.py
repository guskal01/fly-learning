from constants import *
from models import Net

from torch.utils.data import DataLoader

class SimilarModel():
    def __init__(self, stealthiness):
        self.stealthiness = stealthiness

    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.train()
        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            print("Epoch", epoch)
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

        bad_net = Net().to(device)
        bad_net.load_state_dict(net.state_dict())
        bad_opt = torch.optim.Adam(bad_net.parameters(), lr=LR)
        for epoch in range(1, 5):
            print("Bad epoch", epoch)
            for data, target in dataloader:
                data, target = data.to(device), -target.to(device)
                bad_opt.zero_grad()
                output = bad_net(data)
                loss = net.loss_fn(output, target)
                for key in bad_net.state_dict():
                    loss += self.stealthiness * ((bad_net.state_dict()[key] - net.state_dict()[key].detach().clone()) ** 2).sum(dim=None)
                loss.backward()
                bad_opt.step()

        return epoch_train_losses
