import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torch.utils.data.dataloader import default_collate
import pytorch3d.ops
from tqdm.auto import tqdm
import argparse

from .pcl import PointCloudDataset
from .misc import str_list
from .transforms import standard_train_transforms



def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_A.size(0)  # 对于10000精度的点云来说，N=10000,即原始点云中点的个数；
    # torch.randperm(N)将0~N-1随机打乱后获得数字序列，函数名是random permutation的缩写
    seed_idx = torch.randperm(N)[:num_patches]  # (num_patches, )
    # 当num_patches=1，即patch个数是1时，seed_idx是包含一个数一维tensor，所以pcl_A[seed_idx]是2维tensor，shape(1,3)
    # pytorch3d.ops.knn_points返回dists(K近邻到采样点的距离)、idx(K个近邻在原点云中的索引)、nn(K个近邻的坐标,4维tensor(点云数，采样点数，K，3))
    # 在噪声点云中构造某个点的patch
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)  # (1, P, 3)
    _, _, pat_A = pytorch3d.ops.knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
    pat_A = pat_A[0]  # (P, M, 3)

    # 在干净点云中构造某个点的patch，K=1200
    _, idx_B, pat_B = pytorch3d.ops.knn_points(seed_pnts, pcl_B.unsqueeze(0), K=int(ratio * patch_size), return_nn=True)
    idx_B = idx_B[0]  # 二维tensor(P采样点数，K数），因为P=1，就一个采样点，所以(1,1200)
    pat_B = pat_B[0]  # 二维tensor(P采样点数，K数, 点的维度），因为P=1，就一个采样点，所以(1,1200,3)

    return pat_A, pat_B, seed_pnts, seed_idx


class PairedPatchDataset(Dataset):

    def __init__(self, datasets, split='train', patch_size=1000, num_patches=1000, patch_ratio=1.0, on_the_fly=True,
                 transform=None):
        super().__init__()
        self.datasets = datasets
        self.split = split
        self.len_datasets = sum([len(dset) for dset in datasets])  # 对于三个精度的数据集来说，每个数据集都有40个点云，总共120个
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform
        self.patches = []

    def __len__(self):
        # 对三个精度的数据集来说，该函数返回3(数据集数) * 40(每个数据集的点云数) * 1000(patch数)
        return self.len_datasets * self.num_patches

    def __getitem__(self, idx):

        pcl_dset = random.choice(self.datasets)  # 如果self.datasets是多个分辨率的数据集，则随机选择一个数据集
        pcl_data = pcl_dset[idx % len(pcl_dset)]
        """
        # 某个pcl_data如下
        {
            'pcl_clean': tensor([[-1.5069e-02, -5.2270e-01,  2.5606e-05],
                                [-2.0572e-01,  3.6817e-02,  8.0125e-03],
                                [ 5.0698e-01, -4.0610e-01,  3.6928e-01],
                                ...,
                                [ 7.0666e-01, -7.1298e-02, -1.1280e-01],
                                [-8.4494e-01,  1.9336e-01,  9.6439e-02],
                                [ 6.6091e-01, -7.1789e-02, -1.5756e-01]]), 
            'name': 'armadillo', 
            'center': tensor([[-0.0239, -0.0989,  0.0210]]), 
            'scale': tensor([[0.9335]]), 
            'pcl_noisy': tensor([[ 0.0252, -0.3032,  0.0574],
                                [-0.1562,  0.0695, -0.0680],
                                [ 0.7618, -0.4637,  0.3043],
                                ...,
                                [ 0.8207, -0.1387, -0.1618],
                                [-0.8659,  0.2513,  0.2407],
                                [ 1.0071, -0.0789,  0.1306]]), 
            'noise_std': 0.1141432089715423
        }
        """

        if self.split == 'train':
            # print("pcl_data['pcl_noisy'].shape", pcl_data['pcl_noisy'].shape)
            # print("pcl_data['name']", pcl_data['name'])

            # 将噪声点云和干净点云数据进去，构造一个噪声patch和一个干净patch、以及选取的噪声点坐标、噪声点索引
            pat_noisy, pat_clean, seed_pts, seed_idx = make_patches_for_pcl_pair(
                pcl_data['pcl_noisy'],
                pcl_data['pcl_clean'],
                patch_size=self.patch_size,
                num_patches=1,
                ratio=self.patch_ratio
            )
            # pat_noisy，在噪声点云中构造的patch，shape(num_patches 1, patch_size, 3)
            # pat_clean，在干净点云中构造的patch，shape(num_patches 1, ratio * patch_size, 3)
            # seed_pts，在噪声点云中选的采样点（即每个patch的中心），shape(1, num_patches 1, 3)
            # seed_idx，seed_pts在噪声点云中的下标，tensor数
            pat_std = pcl_data['noise_std']
            # data['pcl_noisy'].shape(1000, 3)
            # data['pcl_clean'].shape(1200, 3)
            # data['seed_pnts'].shape(1, 3)
            # data['pcl_std']是个tensor数
            data = {
                'pcl_noisy': pat_noisy[0],
                'pcl_clean': pat_clean[0],
                'seed_pnts': seed_pts[0],
                'pcl_std': pat_std
            }
            # data的一个实例：
            """
            {
                'pcl_noisy': tensor([[ 0.0808, -0.8173, -0.2372],
                                     [ 0.0971, -0.8068, -0.2213],
                                     [ 0.0981, -0.7983, -0.2412],
                                        ...,
                                     [-0.1368, -0.6592, -0.0462],
                                     [-0.1365, -0.7591,  0.0046],
                                     [ 0.0758, -0.5812, -0.4680]]), 
                'pcl_clean': tensor([[-0.0138, -0.7945, -0.2364],
                                     [-0.0120, -0.7882, -0.2472],
                                     [-0.0162, -0.7972, -0.2535],
                                        ...,
                                     [-0.1365, -0.6053, -0.3384],
                                     [ 0.0214, -0.5032, -0.2184],
                                     [-0.2009, -0.6662, -0.2562]]), 
                'seed_pnts': tensor([[ 0.0808, -0.8173, -0.2372]]), 
                'pcl_std': 0.1885447963928971
            }
            """
        ### 以下代码没有
        else:
            pat_noisy, pat_clean, seed_pts, seed_idx = make_patches_for_pcl_pair(
                pcl_data['pcl_noisy'],
                pcl_data['pcl_clean'],
                None,
                patch_size=self.patch_size,
                num_patches=1,
                ratio=self.patch_ratio
            )

            data = {
                'pcl_noisy': pat_noisy[0],
                'pcl_clean': pat_clean[0],
            }

        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'], choices=[['x0000_poisson'], ['x0000_poisson', 'y0000_poisson'], ['x0000_poisson', 'y0000_poisson', 'z0000_poisson']])
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--patches_per_shape_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_ratio', type=float, default=1.2)


    args = parser.parse_args()
    args.batch_size = 32

    print(args)
    train_dset = PairedPatchDataset(
        datasets = [
            PointCloudDataset(
                root=args.dataset_root,
                dataset=args.dataset,
                split='train',
                resolution=resl,
                transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min, rotate=args.aug_rotate)
            ) for resl in args.resolutions
        ],
        split='train',
        patch_size=args.patch_size,
        num_patches=args.patches_per_shape_per_epoch,
        patch_ratio=args.patch_ratio,
        on_the_fly=True  # Currently, we only support on_the_fly=True
    )

    print(train_dset[0]['pcl_noisy'].shape)
    print(train_dset[0]['pcl_clean'].shape)


