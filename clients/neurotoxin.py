from constants import *

from torch.utils.data import DataLoader

class NeurotoxinAttack():
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
                
                for param_tensor in net.state_dict():
                    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

                batch_train_losses.append(loss.item())
            epoch_train_losses.append(sum(batch_train_losses)/len(batch_train_losses))
        return epoch_train_losses
    
    