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

class CClipDefense():
    def __init__(self, dataloader, n_attackers):
        self.dataloader = dataloader
        self.tau = 10
        self.bucket_size = 5
        self.initial_guess = None
    
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
        return net


    def defend_before_aggregation(
            self,
            raw_client_grad_list,
            extra_auxiliary_info = None,
    ):
        client_grad_buckets = Bucket.bucketization(
            raw_client_grad_list, self.bucket_size
        )
        self.initial_guess = self._compute_an_initial_guess(client_grad_buckets)
        bucket_num = len(client_grad_buckets)
        vec_local_w = [
            (
                client_grad_buckets[i][0],
                self.vectorize_weight(client_grad_buckets[i][1]),
            )
            for i in range(bucket_num)
        ]
        vec_refs = self.vectorize_weight(self.initial_guess)
        cclip_score = self._compute_cclip_score(vec_local_w, vec_refs)
        new_grad_list = []
        for i in range(bucket_num):
            tuple = OrderedDict()
            sample_num, bucket_params = client_grad_buckets[i]
            for k in bucket_params.keys():
                tuple[k] = (bucket_params[k] - self.initial_guess[k]) * cclip_score[i]
            new_grad_list.append((sample_num, tuple))
        return new_grad_list

    def defend_after_aggregation(self, global_model):
        for k in global_model.keys():
            global_model[k] = self.initial_guess[k] + global_model[k]
        return global_model

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

    @staticmethod
    def _compute_an_initial_guess(client_grad_list):
        # randomly select a gradient as the initial guess
        return client_grad_list[np.random.randint(0, len(client_grad_list))][1]

    def _compute_cclip_score(self, local_w, refs):
        cclip_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dist = self.compute_euclidean_distance(local_w[i][1], refs).item() + 1e-8
            score = min(1, self.tau / dist)
            cclip_score.append(score)
        return cclip_score