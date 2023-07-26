import torch
from constants import *

class Krum():
    def __init__(self, dataloader, n_attackers):
        self.dataloader = dataloader
        self.byzantine_client_num = n_attackers
        self.krum_param_m = 4
    
    def aggregate(self, net, client_nets, selected):
        model_list = []
        for i,x in enumerate(client_nets):
            for key in net.state_dict():
                x[key] -= net.state_dict()[key]
            model_list.append((i, x))
        result = self.defend_before_aggregation(model_list)

        state_dict = net.state_dict()
        
        for key in state_dict:
            state_dict[key] = sum([x[1][key] for x in result]) / len(result) + state_dict[key]
        
        net.load_state_dict(state_dict)
        return net, None

    def defend_before_aggregation(
        self,
        raw_client_grad_list,
    ):
        num_client = len(raw_client_grad_list)
        # in the Krum paper, it says 2 * byzantine_client_num + 2 < client #
        if not 2 * self.byzantine_client_num + 2 <= num_client - self.krum_param_m:
            raise ValueError(
                "byzantine_client_num conflicts with requirements in Krum: 2 * byzantine_client_num + 2 < client number - krum_param_m"
            )

        vec_local_w = [
            self.vectorize_weight(raw_client_grad_list[i][1])
            for i in range(0, num_client)
        ]
        krum_scores = self._compute_krum_score(vec_local_w)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0 : self.krum_param_m]
        return [raw_client_grad_list[i] for i in score_index]

    def is_weight_param(self, k):
        return (
                "running_mean" not in k
                and "running_var" not in k
                and "num_batches_tracked" not in k
        )

    def vectorize_weight(self, state_dict):
        weight_list = []
        for (k, v) in state_dict.items():
            if self.is_weight_param(k):
                weight_list.append(v.flatten())
        return torch.cat(weight_list)

    def compute_euclidean_distance(self, v1, v2, device='cpu'):
        v1 = v1.to(device)
        v2 = v2.to(device)
        return (v1 - v2).norm()

    def _compute_krum_score(self, vec_grad_list):
        krum_scores = []
        num_client = len(vec_grad_list)
        for i in range(0, num_client):
            dists = []
            for j in range(0, num_client):
                if i != j:
                    dists.append(
                        self.compute_euclidean_distance(
                            vec_grad_list[i], vec_grad_list[j]
                        ).item() ** 2
                    )
            dists.sort()  # ascending
            score = dists[0 : num_client - self.byzantine_client_num - 2]
            krum_scores.append(sum(score))
        return krum_scores
