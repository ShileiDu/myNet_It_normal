import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from plyfile import PlyData
import argparse

from .transforms import standard_train_transforms
from .misc import str_list


class PointCloudDataset(Dataset):
    """
    root/dataset/pointclouds/split/resolution文件夹下有多少个点云，这个数据集的长度就是几
    对数据集进行索引时，取到的某一个完整点云，然后对其进行一些归一化和添加噪声的操作
    """

    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        self.resolution = resolution
        self.split = split
        # 如，self.pcl_dir = ./data/PUNet/pointclouds/train/10000_poisson
        self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))

            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)  # 某个文件夹下点云的个数，如./data/PUNet/pointclouds/train/10000_poisson文件夹下点云的个数为40

    def __str__(self):
        return "Dataset with resolution: {}".format(self.resolution)  # 返回这个个文件夹点云的分辨率，如10000_poisson

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)

        """
        data的一个实例，if self.transform is None:
        {
            'pcl_clean': tensor([[-1.5069e-02, -5.2270e-01,  2.5606e-05],
                                [-2.0572e-01,  3.6817e-02,  8.0125e-03],
                                [ 5.0698e-01, -4.0610e-01,  3.6928e-01],
                                ...,
                                [ 7.0666e-01, -7.1298e-02, -1.1280e-01],
                                [-8.4494e-01,  1.9336e-01,  9.6439e-02],
                                [ 6.6091e-01, -7.1789e-02, -1.5756e-01]]), 
            'name': 'armadillo'
        }
        data的一个实例，if self.transform is not None:
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


    args = parser.parse_args()
    args.batch_size = 32

    print(args)
    aDataset = PointCloudDataset(
        root=args.dataset_root,
        dataset=args.dataset,
        split='train',
        resolution=args.resolutions[0],
        transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min, rotate=args.aug_rotate)
    )

    print(aDataset[0])
