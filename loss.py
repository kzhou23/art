import torch
from kaolin.metrics.point import SidedDistance

def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out
    
def chamfer_distance(s1, s2, w1=1., w2=1.):
    """
    :param s1: B x N x 3
    :param s2: B x M x 3
    :param w1: weight for distance from s1 to s2
    :param w2: weight for distance from s2 to s1
    """
    assert s1.is_cuda and s2.is_cuda
    sided_minimum_dist = SidedDistance()
    closest_index_in_s2 = sided_minimum_dist(s1, s2)
    closest_index_in_s1 = sided_minimum_dist(s2, s1)
    closest_s2 = batch_gather(s2, closest_index_in_s2)
    closest_s1 = batch_gather(s1, closest_index_in_s1)
    dist_to_s2 = (((s1 - closest_s2) ** 2).sum(dim=-1)).mean() * w1
    dist_to_s1 = (((s2 - closest_s1) ** 2).sum(dim=-1)).mean() * w2
    return dist_to_s2 + dist_to_s1