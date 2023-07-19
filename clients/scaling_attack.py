from constants import *

from torch.utils.data import DataLoader

class ScalingAttack():
    def __init__(self):
        pass

    def train_client(self, net, opt, dataset):
        dataloader = DataLoader(dataset, batch_size=32)

        net.train()
        epoch_train_losses = []
        for epoch in range(1, EPOCHS_PER_ROUND+1):
            batch_train_losses = []
            for data, target in dataloader:
                # Set target to something crazy unreasonable
                data, target = data.to(device), -100*target.to(device)
                opt.zero_grad()
                output = net(data)
                loss = net.loss_fn(output, target)
                loss.backward()
                opt.step()

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        
        state_dict = net.state_dict()
        for key in state_dict:
            state_dict[key] *= 10
        
        net.load_state_dict(state_dict)

        return epoch_train_losses
