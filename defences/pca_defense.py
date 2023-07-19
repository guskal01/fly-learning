from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt 
from models import Net
from constants import *
import torch
import random

def state_dict_to_vec_linear(state_dict):
    weight_keys = [
        "model.classifier.0.weight",
        "model.classifier.3.0.weight",
        "model.classifier.3.2.weight",
        "model.classifier.3.4.weight"
    ]
    
    bias_keys = [
        "model.classifier.0.bias",
        "model.classifier.3.0.bias",
        "model.classifier.3.2.bias",
        "model.classifier.3.4.bias"
    ]

    layers = [weight_keys[-1]]
    #layers = weight_keys + bias_keys
    
    return torch.cat([torch.flatten(state_dict[key]) for key in state_dict if key in layers])

class PCADefense():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.random_idxs = None
    
    def aggregate(self, net, client_nets, selected):
        state_dict = net.state_dict()

        if (self.random_idxs is None):
            self.set_random_param_idxs(client_nets)

        client_params = []
        for i in range(len(client_nets)):
            params = list(state_dict_to_vec_linear(client_nets[i]).detach().cpu())
            client_params.append([params[idx] for idx in self.random_idxs])

        pca = PCA(n_components=2)
        pca.fit(client_params)
        transformed_points = pca.transform(client_params)

        selected_idxs = self.euclidean_distance_remove(transformed_points, 2, plot=True)
        #selected_idxs = self.isolation_forest(transformed_points)
        selected_client_nets = [client_nets[i] for i in selected_idxs]

        for key in state_dict:
            state_dict[key] = sum([x[key] for x in selected_client_nets]) / len(selected_client_nets)
        
        net.load_state_dict(state_dict)
        return net

    def set_random_param_idxs(self, client_nets):
        # Could group parameters in group of 10
        self.random_idxs = range(len(list(state_dict_to_vec_linear(client_nets[0]).detach().cpu())))
        random.shuffle(list(self.random_idxs))
        self.random_idxs = self.random_idxs[:10]

    def euclidean_distance_remove(self, points, k, plot=False):
        distances = []
        for i in range(len(points)):
            dist = 0
            for j in range(len(points)):
                dist += (points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2
            distances.append([dist, i])
        distances.sort()

        if (plot):
            plt.scatter([points[d[1]][0] for d in distances[:-k]], [points[d[1]][1] for d in distances[:-k]])
            plt.scatter([points[d[1]][0] for d in distances[-k:]], [points[d[1]][1] for d in distances[-k:]])
            plt.legend(labels=["Included", "Removed"])
            plt.savefig("./results/out.png")
            plt.clf()

        return [d[1] for d in distances[:-k]]

    def isolation_forest(self, points, plot=False):
        clf = IsolationForest(n_estimators=10, warm_start=True)
        clf.fit(points)
        outliers = clf.predict(points)

        if (plot):
            plt.scatter([p[0] for i, p in enumerate(points) if outliers[i] == 1], [p[1] for i, p in enumerate(points) if outliers[i] == 1])
            plt.scatter([p[0] for i, p in enumerate(points) if outliers[i] == -1], [p[1] for i, p in enumerate(points) if outliers[i] == -1])
            plt.savefig("./results/out.png")
            plt.clf()

        return [i for i, o in enumerate(outliers) if o == 1]

if __name__ == "__main__":
    net = Net().to(device)
    clients = [Net().to(device).state_dict() for _ in range(10)]
    selected = None 

    defense = PCADefense(None)
    defense.aggregate(net, clients, selected)