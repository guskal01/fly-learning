import torch
from constants import *
import sklearn.metrics.pairwise as smp
import numpy as np

class FoolsGoldDefense():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.memory = None
    
    def aggregate(self, net, client_nets, selected):
        model_list = []
        nbt = None
        for i,x in enumerate(client_nets):
            for key in net.state_dict():
                x[key] -= net.state_dict()[key]
                if 'num_batches_tracked' in key:
                    if nbt is not None: assert x[key] == nbt
                    nbt = x.pop(key)
            model_list.append((i, x))
        result, alphas = self.defend_before_aggregation(model_list)

        state_dict = net.state_dict()
        
        for key in state_dict:
            if 'num_batches_tracked' in key:
                state_dict[key] = nbt
            else:
                state_dict[key] = sum([x[1][key] for x in result]) / len(result) + state_dict[key]
        
        net.load_state_dict(state_dict)
        return net, alphas

    def defend_before_aggregation(
        self,
        raw_client_grad_list,
        extra_auxiliary_info = None,
    ):
        client_num = len(raw_client_grad_list)
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        # print(len(importance_feature_list))

        if self.memory is None:
            self.memory = importance_feature_list
        else:  # memory: potential bugs: grads in different iterations may be from different clients
            for i in range(client_num):
                self.memory[i] += importance_feature_list[i]
        alphas = self.fools_gold_score(self.memory)  # Use FG

        print("alphas = {}".format(alphas))
        assert len(alphas) == len(
            raw_client_grad_list
        ), "len of wv {} is not consistent with len of client_grads {}".format(len(alphas), len(raw_client_grad_list))
        new_grad_list = []
        client_num = len(raw_client_grad_list)
        for i in range(client_num):
            sample_num, grad = raw_client_grad_list[i]
            new_grad_list.append((sample_num * alphas[i] / client_num, grad))
        return new_grad_list, alphas

    # Takes in grad, compute similarity, get weightings
    @classmethod
    def fools_gold_score(cls, feature_vec_list):
        n_clients = len(feature_vec_list)
        cs = smp.cosine_similarity(feature_vec_list) - np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        alpha = 1 - (np.max(cs, axis=1))
        alpha[alpha > 1.0] = 1.0
        alpha[alpha <= 0.0] = 1e-15

        # Rescale so that max value is alpha
        # print(np.max(alpha))
        alpha = alpha / np.max(alpha)
        alpha[(alpha == 1.0)] = 0.999999

        # Logit function
        alpha = np.log(alpha / (1 - alpha)) + 0.5
        alpha[(np.isinf(alpha) + alpha > 1)] = 1
        alpha[(alpha < 0)] = 0

        return alpha

    def _get_importance_feature(self, raw_client_grad_list):
        # Foolsgold uses the last layer's gradient/weights as the importance feature.
        ret_feature_vector_list = []
        for idx in range(len(raw_client_grad_list)):
            (p, grads) = raw_client_grad_list[idx]

            # Get last key-value tuple
            (weight_name, importance_feature) = list(grads.items())[-2]
            # print(importance_feature)
            feature_len = np.array(importance_feature.cpu().data.detach().numpy().shape).prod()
            feature_vector = np.reshape(importance_feature.cpu().data.detach().numpy(), feature_len)
            ret_feature_vector_list.append(feature_vector)
        return ret_feature_vector_list