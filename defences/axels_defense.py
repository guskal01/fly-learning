from skimage.metrics import structural_similarity
from scipy import spatial

import matplotlib.pyplot as plt

class AxelsDefense():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.weight_keys = [
            "model.classifier.0.weight",
            "model.classifier.3.0.weight",
            "model.classifier.3.2.weight",
            "model.classifier.3.4.weight"
        ]
        self.bias_keys = [
            "model.classifier.0.bias",
            "model.classifier.3.0.bias",
            "model.classifier.3.2.bias",
            "model.classifier.3.4.bias"
        ]
        self.stats = []
    
    def aggregate(self, net, client_nets, selected):
        self.stats.append(self.get_stats(net, client_nets))
        state_dict = net.state_dict()
        
        for key in state_dict:
            state_dict[key] = sum([x[key] for x in client_nets]) / len(client_nets)
        
        net.load_state_dict(state_dict)
        return net, None

    def get_stats(self, net, client_nets):
        state_dict = net.state_dict()

        metrics = [[] for _ in range(len(client_nets))]
        for key in self.weight_keys:
            print(f"Metrics for key: {key}")
            for i, client_net in enumerate(client_nets):
                ssi, cosine_sim = self.metrics(client_net[key], state_dict[key])
                metrics[i].append([ssi, cosine_sim]) 
        
        return metrics

    def metrics(self, w1, w2):
        w1 = w1.cpu().detach().numpy().flatten().astype(float)
        w2 = w2.cpu().detach().numpy().flatten().astype(float)
        ssi = structural_similarity(w1, w2, data_range=100.0)

        cosine_sim = 1 - spatial.distance.cosine(w1.flatten(), w2.flatten())
        return ssi, cosine_sim

    def plot_stats(self, path):
        ssi_data = [[] for _ in range(len(self.stats[0]))]
        cosine_data = [[] for _ in range(len(self.stats[0]))]
        for global_round in range(len(self.stats)):
            for client_idx in range(len(self.stats[global_round])):
                client_metrics = self.stats[global_round][client_idx][0]
                client_ssi = client_metrics[0]
                client_cosine = client_metrics[1]
                ssi_data[client_idx].append(client_ssi)
                cosine_data[client_idx].append(client_cosine)
        
        for i in range(len(ssi_data)):
            print(f"SSI for client {i}:", ssi_data[i])

        print()

        for i in range(len(cosine_data)):
            print(f"Cosine sim for client{i}", cosine_data[i])