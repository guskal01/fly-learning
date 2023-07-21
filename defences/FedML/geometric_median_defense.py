import math
from collections import OrderedDict
import torch
import numpy as np

class Bucket:
    @classmethod
    def bucketization(cls, client_grad_list, batch_size):
        (num0, averaged_params) = client_grad_list[0]
        batch_grad_list = []
        for batch_idx in range(0, math.ceil(len(client_grad_list) / batch_size)):
            client_num = cls._get_client_num_current_batch(
                batch_size, batch_idx, client_grad_list
            )
            sample_num = cls._get_total_sample_num_for_current_batch(
                batch_idx * batch_size, client_num, client_grad_list
            )
            batch_weight = OrderedDict()
            for i in range(0, client_num):
                local_sample_num, local_model_params = client_grad_list[
                    batch_idx * batch_size + i
                ]
                w = local_sample_num / sample_num
                for k in averaged_params.keys():
                    if i == 0:
                        batch_weight[k] = local_model_params[k] * w
                    else:
                        batch_weight[k] += local_model_params[k] * w
            batch_grad_list.append((sample_num, batch_weight))
        return batch_grad_list

    @staticmethod
    def _get_client_num_current_batch(batch_size, batch_idx, client_grad_list):
        current_batch_size = batch_size
        # not divisible
        if (
            len(client_grad_list) % batch_size > 0
            and batch_idx == math.ceil(len(client_grad_list) / batch_size) - 1
        ):
            current_batch_size = len(client_grad_list) - (batch_idx * batch_size)
        return current_batch_size

    @staticmethod
    def _get_total_sample_num_for_current_batch(
        start, current_batch_size, client_grad_list
    ):
        training_num_for_batch = 0
        for i in range(0, current_batch_size):
            local_sample_number, local_model_params = client_grad_list[start + i]
            training_num_for_batch += local_sample_number
        return training_num_for_batch

class GeometricMedianDefense():
    def __init__(self, config):
        self.byzantine_client_num = config.byzantine_client_num
        self.client_num_per_round = config.client_num_per_round
        # 2(1 + ε )q ≤ batch_num ≤ client_num_per_round
        # trade-off between accuracy & robustness:
        #       larger batch_num --> more Byzantine robustness, larger estimation error.
        self.batch_num = config.batch_num
        if self.byzantine_client_num == 0:
            self.batch_num = 1
        self.batch_size = math.ceil(self.client_num_per_round / self.batch_num)

    def __init__(self, dataloader, n_attackers):
        self.dataloader = dataloader
        self.byzantine_client_num = n_attackers
        self.krum_param_m = 4
        self.batch_size = 64
    
    def aggregate(self, net, client_nets, selected):
        model_list = []
        for i,x in enumerate(client_nets):
            for key in net.state_dict():
                x[key] -= net.state_dict()[key]
            model_list.append((i, x))
        result = self.defend_on_aggregation(model_list)

        state_dict = net.state_dict()
        
        for key in state_dict:
            state_dict[key] = sum([x[1][key] for x in result]) / len(result) + state_dict[key]
        
        net.load_state_dict(state_dict)
        return net



    def defend_on_aggregation(
            self,
            raw_client_grad_list,
            base_aggregation_func = None,
            extra_auxiliary_info = None,
    ):
        batch_grad_list = Bucket.bucketization(raw_client_grad_list, self.batch_size)
        (num0, avg_params) = batch_grad_list[0]
        alphas = {alpha for (alpha, params) in batch_grad_list}
        alphas = {alpha / sum(alphas, 0.0) for alpha in alphas}
        for k in avg_params.keys():
            batch_grads = [params[k] for (alpha, params) in batch_grad_list]
            avg_params[k] = self.compute_geometric_median(alphas, batch_grads)
        return avg_params

    def compute_middle_point(self, alphas, model_list):
        """
        alphas: weights of model_dict
        model_dict: a model submitted by a user
        """
        sum_batch = torch.zeros(model_list[0].shape)
        for a, a_batch_w in zip(alphas, model_list):
            sum_batch += a * a_batch_w.float().cpu().numpy()
        return sum_batch
    
    def compute_euclidean_distance(self, v1, v2, device='cpu'):
        v1 = v1.to(device)
        v2 = v2.to(device)
        return (v1 - v2).norm()


    def compute_geometric_median(self, weights, client_grads):
        """
        Implementation of Weiszfeld's algorithm.
        Reference:  (1) https://github.com/krishnap25/RFA/blob/master/models/model.py
                    (2) https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/geomed.py
        our contribution: (07/01/2022)
        1) fix one bug in (1): (1) can not correctly compute a weighted average. The function weighted_average_oracle
        returns zero.
        2) fix one bug in (2): (2) can not correctly handle multidimensional tensors.
        3) reconstruct the code.
        """
        eps = 1e-5
        ftol = 1e-10
        middle_point = self.compute_middle_point(weights, client_grads)
        val = sum(
            [
                alpha * self.compute_euclidean_distance(middle_point, p)
                for alpha, p in zip(weights, client_grads)
            ]
        )
        for i in range(100):
            prev_median, prev_obj_val = middle_point, val
            weights = np.asarray(
                [
                    max(
                        eps,
                        alpha
                        / max(eps, self.compute_euclidean_distance(middle_point, a_batch_w)),
                    )
                    for alpha, a_batch_w in zip(weights, client_grads)
                ]
            )
            weights = weights / weights.sum()
            middle_point = self.compute_middle_point(weights, client_grads)
            val = sum(
                [
                    alpha * self.compute_euclidean_distance(middle_point, p)
                    for alpha, p in zip(weights, client_grads)
                ]
            )
            if abs(prev_obj_val - val) < ftol * val:
                break
        return middle_point