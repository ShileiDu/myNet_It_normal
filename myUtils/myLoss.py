import torch
import pytorch3d.ops


def get_supervised_loss_nn(pcl_noisy, pred_disp, pcl_clean):
    """
    Denoising score matching.
    Args:
        pcl_noisy:  Noisy point clouds, (B, N1000, 3).输入的噪声patch
        pred_disp: 预测的位移
        pcl_clean:  Clean point clouds, (B, M1200, 3).输入的干净点云patch
    """
    # B=arg.train_batch_size=4
    # N_noisy，一个噪声patch包含的点数，即patch size, 1000
    # N_clean，一个干净patch包含的点数，即ratio*patch_size, 1200
    B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
    # B32, N_noisy1000, N_clean1200, d3

    # 计算每个patch中近邻点到种子点（移到中心了，所以是0，0，0）距离对采样点的权重
    seed_dist_sq = (pcl_noisy ** 2).sum(dim=-1, keepdim=True)  # (批大小，patch size，1),(4, 1000, 1)
    # 因为knn添加点的时候是从最近的点添加的，所以每个patch最后一个添加的点的距离最大
    max_seed_dist_sq = seed_dist_sq[:, -1, :]  # shape(批大小，1)
    seed_dist_sq = seed_dist_sq / (max_seed_dist_sq.unsqueeze(1) / 9)  # shape(批大小,patch size,1)
    seed_weights = torch.exp(-1 * seed_dist_sq).squeeze()  # shape(批大小, patch size)
    seed_weights_sum = seed_weights.sum(dim=1, keepdim=True)  # shape(批大小，1）
    seed_weights = (seed_weights / seed_weights_sum).squeeze()  # squeeze()前后的shape都是(批大小，patch size)

    _, _, clean_pts = pytorch3d.ops.knn_points(
        pcl_noisy,  # (B, N, 3)
        pcl_clean,  # (B, M, 3)
        K=1,
        return_nn=True,
    )  # (B, N, K, 3), (4, 1000, 1, 3)
    #clean_pts.shape(32, 1000, 1, 3)
    clean_nbs = clean_pts.view(B, N_noisy, d)  # (B, N, 3), (4, 1000, 3)

    clean_nbs = clean_nbs - pcl_noisy
    dist = ((pred_disp - clean_nbs) ** 2).sum(dim=-1)  # (B, N)
    losses= (seed_weights * dist).sum(dim=-1).mean(dim=-1)

    return losses


if __name__ == "__main__":
    pcl_noisy = torch.rand((32, 1000, 3)).float().cuda()
    pred_disp = torch.rand((32, 1000, 3)).float().cuda()
    pcl_clean = torch.rand((32, 1200, 3)).float().cuda()
    get_supervised_loss_nn(pcl_noisy, pred_disp, pcl_clean)