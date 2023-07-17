from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt 

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

    
    return torch.cat([torch.flatten(state_dict[key]) for key in state_dict if key in weight_keys+bias_keys])

class PCADefense():
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def aggregate(self, net, client_nets, selected):
        state_dict = net.state_dict()
        

        client_params = []
        for i in range(len(client_nets)): 
            #print(state_dict_to_vec(client_nets[i]).detach().cpu())
            client_params.append(list(state_dict_to_vec_linear(client_nets[i]).detach().cpu()))

        pca = PCA(n_components=2)
        pca.fit(client_params)
        x = pca.transform(client_params)

        clf = IsolationForest(n_estimators=10, warm_start=True)
        clf.fit(x) 
        outliers = clf.predict(x) 

        print(x.shape)
        print(x)
        print(outliers)

        plt.scatter([p[0] for p in x], [p[1] for p in x])
        plt.savefig("out.png")
        assert 1==0

        for key in state_dict:
            state_dict[key] = sum([x[key] for x in client_nets]) / len(client_nets)
        
        net.load_state_dict(state_dict)
        return net


if __name__ == "__main__":
    from models import Net
    from constants import *

    net = Net().to(device)
    clients = [Net().to(device).state_dict() for _ in range(10)]
    selected = None 

    defense = PCADefense(None)
    defense.aggregate(net, clients, selected)