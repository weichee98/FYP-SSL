import torch
import numpy as np
from abc import ABC, abstractmethod
from captum.attr import IntegratedGradients
from scipy.spatial.distance import squareform


class SaliencyScoreForward(ABC):
    @abstractmethod
    def ss_forward(self, *args):
        raise NotImplementedError

    def get_baselines_inputs(self, data):
        x, y = data.x, data.y
        baselines = x[y == 0].mean(dim=0).view(1, -1)
        inputs = x[y == 1]
        return baselines, inputs

    def saliency_score(self, data):
        baselines, inputs = self.get_baselines_inputs(data)
        ig = IntegratedGradients(self.ss_forward, True)
        scores = ig.attribute(inputs=inputs, baselines=baselines, target=1)

        scores = scores.detach().cpu().numpy()
        scores = np.array([squareform(score) for score in scores])
        return scores


class GraphSaliencyScoreForward(SaliencyScoreForward):
    def get_baselines_inputs(self, data):
        baselines = [(d.x, d.adj_t) for d in data if torch.all(d.y == 0)]
        baselines_x, baselines_adj = zip(*baselines)
        baselines_x = torch.stack(baselines_x, dim=0).mean(dim=0)
        baselines_adj = torch.stack(baselines_adj, dim=0).mean(dim=0)

        inputs = [d for d in data if torch.all(d.y == 1)]
        return (baselines_x, baselines_adj), inputs

    def get_saliency_score(self, data, baselines):
        x, adj = data.x, data.adj_t
        ig = IntegratedGradients(self.ss_forward, True)
        _, score = ig.attribute(inputs=(x, adj), baselines=baselines, target=1,)
        return score

    def saliency_score(self, data):
        baselines, inputs = self.get_baselines_inputs(data)
        scores = [self.get_saliency_score(d, baselines) for d in inputs]
        scores = torch.stack(scores, dim=0)
        scores = scores.detach().cpu().numpy()
        return scores
