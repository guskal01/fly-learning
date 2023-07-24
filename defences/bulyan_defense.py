import torch
import numpy as np

class BulyanDefense():
    def __init__(self, dataloader,n_attackers):
        self.dataloader = dataloader
        self.faulty_count = n_attackers
    
    def compute_distances(self, client_nets, median_params):
        distances = {}

        for i in range(len(client_nets)):
            client_params = {}
            distance = 0
            for key in client_nets[i]:
                client_params[key] = torch.tensor(client_nets[i][key])
                if key in median_params:
                    median_diff = client_params[key] - median_params[key]
                    distance += torch.norm(median_diff)

            distances[i] = distance
        return distances
    
    def bulyan(self,client_nets, median_params):
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

        print(f"selected list length is {len(selected_set)}")

        return selected_set
    
    def aggregate(self, net, client_nets, selected=None):
        state_dict = net.state_dict()
        median_params = {}
        client_params = {}
        weight_keys = [key for key in state_dict if "weight" in key]
        for key in weight_keys:
            all_params = torch.stack([torch.tensor(client_params[key]) for client_params in client_nets])
            median_params[key] = torch.median(all_params, dim=0).values
        
        selected_set = self.bulyan(client_nets, median_params)
        print(f"Selected Clients are :{selected_set}")
            
        net.load_state_dict(state_dict)
        return net
    


'''class BulyanDefense():
    def __init__(self, dataloader,n_attackers):
        self.dataloader = dataloader
        self.faulty_count = n_attackers
        
    def _agr(
        self,
        client_nets,
        n_clients,
        faulty_count,
        distances=None,
        return_index=False,
    ):
        non_malicious_count = n_clients - faulty_count
        minimal_error = 1e20
        minimal_error_index = -1
        
        print(f"client net[0]: {client_nets[0]}, type: {type(client_nets[0])}")
        
        if distances is None:
            distances = self.compute_distances(client_nets)
        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            if current_error < minimal_error:
                minimal_error = current_error
                minimal_error_index = user

        if return_index:
            return minimal_error_index
        else:
            return client_nets[minimal_error_index]

    @staticmethod
    def compute_distances(client_nets):
        distances = {}
        state_dict =client_nets[0].state_dict()        
        for key in state_dict:
            if ("bias" in key or "weight" in key):
                for i in range(len(client_nets)):
                    client_params=torch.from_numpy(np.array(client_nets[i][key].cpu().numpy()))
                    sorted_params = client_params.sort(dim=0)[0]
                    distances = client_nets[i] - sorted_params.median(dim=0)
        return distances
    
    def aggregate(self,net,client_nets,selected=None):
        state_dict = net.state_dict() 
        aggregated_params = {}
        index_bias = 0       
        for key in state_dict:
            if ("weight" in key):
                for i in range(len(client_nets)):
                    params=torch.from_numpy(np.array(client_nets[i][key].cpu().numpy()))
                    _,_, agg_grads = self.bulyan(params,len(client_nets), self.faulty_count)
                    aggregated_params[key] = torch.from_numpy(agg_grads[index_bias : index_bias + params.numel()]).view(params.size())  # todo: gpu/cpu issue for torch
                    index_bias += params.numel()
            else:
                aggregated_params[key] = params
            return aggregated_params

        state_dict = net.state_dict()
        return net

    def bulyan(self, client_nets, n_clients, faulty_count):
        assert n_clients >= (4 * faulty_count + 3)-1 #need to check with formula
        set_size = n_clients - 2 * faulty_count
        selection_set = []
        select_indexs = []
        distances = self.compute_distances(client_nets)
        print(distances)

        while len(selection_set) < set_size:
            currently_selected = self._agr(
                client_nets,
                n_clients - len(selection_set),
                faulty_count,
                distances,
                return_index=True,
            )
        
            selection_set.append(client_nets[currently_selected])
            select_indexs.append(currently_selected)
            distances.pop(currently_selected)
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected)

        agg_grads = self.trimmed_mean(selection_set, 2 * faulty_count)

        return select_indexs, selection_set, agg_grads
        '''