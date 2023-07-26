import torch
import numpy as np

class BulyanDefense():
    def __init__(self, dataloader, n_attackers):
        self.dataloader = dataloader
        self.faulty_count = n_attackers
    
    def compute_distances(self, client_nets, median_params):
        distances = {}

        for i in range(len(client_nets)):
            client_params = {}
            distance = 0
            for key in client_nets[i]:
                client_params[key] = client_nets[i][key]
                if key in median_params:
                    median_diff = client_params[key] - median_params[key]
                    distance += torch.norm(median_diff)

            distances[i] = distance
        return distances
    
    def bulyan(self,client_nets, median_params, faulty_count):
        num_clients = len(client_nets)
        theta = num_clients - 2 * self.faulty_count

        # Initialize selected_set and select_indexs
        selected_set = []
        select_indexs = []
        distances = self.compute_distances(client_nets,median_params)
        
        for i in range(theta):
            min_distance = float('inf')
            min_distance_client = None

            # Find the client with the minimum distance
            for i in range(len(distances)):
                if i not in select_indexs:
                    if distances[i] < min_distance:
                        min_distance = distances[i]
                        min_distance_client = i

            # Add the selected client to the selected_set and select_indexs
            selected_set.append(min_distance_client)
            select_indexs.append(min_distance_client)
        
        beta = theta - 2
        final_beta_set = []
        s_distances = {}
        for i in selected_set:
            # Calculate the median parameters for the selected client
            median_params_i = {}
            for key in median_params:
                all_params = torch.stack([client_nets[i][key] for i in selected_set])
                median_params_i[key] = torch.median(all_params, dim=0).values

            # Compute the distances for the selected client to the median parameters
            client_params = {}
            distance = 0
            for key in client_nets[i]:
                client_params[key] = client_nets[i][key]
                if key in median_params:
                    median_diff = client_params[key] - median_params[key]
                    distance += torch.norm(median_diff)

            s_distances[i] = distance

        # Select the beta number of clients with the least distance
        beta_clients = sorted(s_distances.items(), key=lambda x: x[1])[:beta]
        s_distances=[x[1].item() for x in beta_clients]
        beta_clients = [client[0] for client in beta_clients]
        final_beta_set.extend(beta_clients)

        return final_beta_set, s_distances, beta

    def final_aggregate(self,net, selected_client_nets):
        state_dict = net.state_dict()
        
        for key in state_dict:
            state_dict[key] = sum([x[key] for x in selected_client_nets]) / len(selected_client_nets)
        
        net.load_state_dict(state_dict)
        return net      
    
    def aggregate(self, net, client_nets, selected=None):
        state_dict = net.state_dict()
        median_params = {}
        client_params = {}
        weight_keys = [key for key in state_dict if "weight" in key]
        for key in weight_keys:
            all_params = torch.stack([client_params[key] for client_params in client_nets])
            median_params[key] = torch.median(all_params, dim=0).values
        
        final_beta_set, distances, beta = self.bulyan(client_nets, median_params, self.faulty_count)
        selected_client_nets=[client_nets[i] for i in final_beta_set]
        net=self.final_aggregate(net,selected_client_nets)
        selected_clients = [1 if i in final_beta_set else 0 for i in range(len(selected))]
        return net, selected_clients
    