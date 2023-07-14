from sklearn.decomposition import PCA
import numpy as np

class PCADefense():
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def aggregate(self, net, client_nets, selected):
        state_dict = net.state_dict()
        

        client_params = []
        for i in range(len(client_nets)): 
            client_params.append([client_nets[i][key].cpu() for key in state_dict][0])

        pca = PCA(n_components=2)
        principal_components = pca.fit(client_params)

        print(principal_components)

        for key in state_dict:
            state_dict[key] = sum([x[key] for x in client_nets]) / len(client_nets)
        
        net.load_state_dict(state_dict)
        return net
