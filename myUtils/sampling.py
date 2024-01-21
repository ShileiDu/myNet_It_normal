import torch
from torch_cluster import fps



def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    # pcl点云包含的点数为：pcls.size(1)；目标点数：num_pnts
    # ratio表示采样率，比真实的采样率会大一点
    ratio = 0.01 + num_pnts / pcls.size(1) #对于Icosahedron这个点云来说，pcls.size(1)=50000
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        # torch_cluster.fps返回采样点在原点云中的索引
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts] # 取前num_pnts个点
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices
