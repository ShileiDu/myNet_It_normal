# from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import pytorch3d.ops
import torch.optim as optim



import argparse
from myUtils.misc import str_list
from myUtils.pcl import PointCloudDataset
from myUtils.patch import PairedPatchDataset
from myUtils.transforms import standard_train_transforms


class pointfilter_encoder(nn.Module):
    def __init__(self, input_dim=3,sym_op='max'):
        super(pointfilter_encoder, self).__init__()
        self.sym_op = sym_op
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        self.activate = nn.ReLU()

    def forward(self, x):

        x = self.activate(self.bn1(self.conv1(x)))
        # x 二维tensor，shape(Batch_size32, num_feature64, patch_size1000)
        net1 = x  # 64

        x = self.activate(self.bn2(self.conv2(x)))
        # x 二维tensor，shape(Batch_size32, num_feature128, patch_size1000)
        net2 = x  # 128

        x = self.activate(self.bn3(self.conv3(x)))
        # x 二维tensor，shape(Batch_size32, num_feature256, patch_size1000)
        net3 = x  # 256

        x = self.activate(self.bn4(self.conv4(x)))
        # x 二维tensor，shape(Batch_size32, num_feature512, patch_size500)
        net4 = x  # 512

        x = self.activate(self.bn5(self.conv5(x)))
        # x 二维tensor，shape(Batch_size32, num_feature1024, patch_size500)
        net5 = x  # 1024

        # if self.sym_op == 'sum':
        #     x = torch.sum(x, dim=-1)
        # else:
        #     x, index = torch.max(x, dim=-1)


        # x 二维tensor，shape(Batch_size32, num_feature1024)

        return x#, index


class pointfilter_decoder(nn.Module):
    def __init__(self):
        super(pointfilter_decoder, self).__init__()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.3)

    def forward(self, x):

        # x.shape(Batch_size32, num_feature1024)
        x = F.relu(self.bn1(self.fc1(x)))
        # x.shape(Batch_size32, num_feature512)
        # x = self.dropout_1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        # x.shape(Batch_size32, num_feature256)
        # x = self.dropout_2(x)

        x = torch.tanh(self.fc3(x))
        # x.shape(Batch_size32, num_feature3)
        return x


class pointfilternet(nn.Module):
    def __init__(self, input_dim=3, sym_op='max'):
        super(pointfilternet, self).__init__()
        self.sym_op = sym_op
        self.input_dim = input_dim

        self.encoder = pointfilter_encoder(self.input_dim, self.sym_op)
        self.decoder = pointfilter_decoder()

    def forward(self, x):
        # 输入x.shape(Batch_size32, num_feature3, patch_size1000)
        batch_size, num_points= x.shape[0], x.shape[-1]
        x = self.encoder(x)
        # x 二维tensor，shape(Batch_size32, num_feature1024, patch_size1000)

        x =x.transpose(2, 1).contiguous()  # 三维tensor，shape(B32, patch_size1000, num_feature1024)
        x = x.view(batch_size * num_points, -1)

        # x.shape=(B32*num_point1000, 1024)

        x = self.decoder(x)

        x = x.view(batch_size, num_points, -1)
        return x#, encoder_feature


class DenoiseNet(nn.Module):
    def __init__(self, num_modules, noise_decay):
        super(DenoiseNet, self).__init__()
        self.num_modules = num_modules
        self.noise_decay = noise_decay
        self.feature_nets = nn.ModuleList()
        for i in range(self.num_modules):
            self.feature_nets.append(pointfilternet().cuda())

    def forward(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        # N_noisy，一个噪声patch包含的点数，即patch size, 1000
        # N_clean，一个干净patch包含的点数，即ratio*patch_size, 1200
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        # self.num_modules = 4
        losses = torch.zeros(self.num_modules)
        # pcl_seeds.shape(4, 1, 3)，批量大小为4，每一批有一个patch中心点
        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)
        # pcl_seeds_1.shape(4, 1000, 3)，批量大小为4，一批的每一个patch中心点复制了patch_size份

        # 计算每个patch中近邻点对采样点的权重
        seed_dist_sq = ((pcl_noisy - pcl_seeds_1)**2).sum(dim=-1, keepdim=True) # (批大小，patch size，1),(4, 1000, 1)
        # 因为knn添加点的时候是从最近的点添加的，所以每个patch最后一个添加的点的距离最大
        max_seed_dist_sq = seed_dist_sq[:, -1, :] # shape(批大小，1)
        seed_dist_sq = seed_dist_sq / (max_seed_dist_sq.unsqueeze(1) / 9)# shape(批大小,patch size,1)
        seed_weights = torch.exp(-1 * seed_dist_sq).squeeze() # shape(批大小, patch size)
        seed_weights_sum = seed_weights.sum(dim=1, keepdim=True) # shape(批大小，1）
        seed_weights = (seed_weights / seed_weights_sum).squeeze() # squeeze()前后的shape都是(批大小，patch size)

        # 把每个patch的中心移到原点
        # pcl_noisy.shape(4, 1000, 3)
        # pcl_seeds_1.shape(4, 1000, 3)
        pcl_noisy = pcl_noisy - pcl_seeds_1 # 把每个噪声patch的中心移到原点
        # pcl_seeds_2.shape(4, 1000, 3)
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2 # 把每个噪声patch的中心移到原点

        curr_std = pcl_std# 一维tensor，长度=批大小，[0.0486, 0.1387, 0.0278, 0.0391]
        for i in range(self.num_modules):

            if i == 0:
                pcl_input = pcl_noisy
            else:
                pcl_input = pcl_input + pred_disp
            pcl_input = pcl_input.transpose(2, 1).contiguous()
            pred_disp= self.feature_nets[i](pcl_input)
            pcl_input = pcl_input.transpose(2, 1).contiguous()

            if self.noise_decay != 1:
                if i < self.num_modules - 1:
                    curr_std = curr_std / self.noise_decay
                    pcl_target_lower_noise = self.curr_iter_add_noise(pcl_clean, curr_std)
                else:
                    curr_std = 0
                    pcl_target_lower_noise = pcl_clean
            else:
                pcl_target_lower_noise = pcl_clean

            _, _, clean_pts = pytorch3d.ops.knn_points(
                pcl_input,    # (B, N, 3)
                pcl_target_lower_noise,   # (B, M, 3)
                K=1,
                return_nn=True,
            )   # (B, N, K, 3), (4, 1000, 1, 3)
            clean_nbs = clean_pts.view(B, N_noisy, d)  # (B, N, 3), (4, 1000, 3)

            clean_nbs = clean_nbs - pcl_input
            dist = ((pred_disp - clean_nbs)**2).sum(dim=-1) # (B, N)
            losses[i] = (seed_weights * dist).sum(dim=-1).mean(dim=-1)

        return losses.sum()  # , target, scores, noise_vecs


    def curr_iter_add_noise(self, pcl_clean, noise_std):
        new_pcl_clean = pcl_clean + torch.randn_like(pcl_clean) * noise_std.unsqueeze(1).unsqueeze(2)
        return new_pcl_clean.float()

    def denoise(self, pcl_noise):
        pass

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    ## Optimizer and scheduler
    parser.add_argument('--network_model_dir', type=str, default='./Summary/Train') # 保存模型的目录
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--manualSeed', type=int, default=3627473, help='manual seed')
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'], choices=[['x0000_poisson'], ['x0000_poisson', 'y0000_poisson'], ['x0000_poisson', 'y0000_poisson', 'z0000_poisson']])
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])

    parser.add_argument('--patches_per_shape_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_ratio', type=float, default=1.2)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--resume', type=str, default='', help='refine model at this path')

    ## train
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parameters = parser.parse_args()
    parameters.nepoch = 3

    # Ablation parameters
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--num_modules', type=int, default=4)
    parser.add_argument('--noise_decay', type=int, default=4)  # Noise decay is set to 16/T where T=num_modules or set to 1 for no decay

    args = parser.parse_args()

    denoisenet = DenoiseNet(args.num_modules, args.noise_decay).cuda()
    # 5.实例化优化器
    # args.lr 1e-2
    # args.momentum 0.9

    optimizer = optim.SGD(
        denoisenet.parameters(),
        lr=args.lr,
        momentum=args.momentum)


    train_dset = PairedPatchDataset(
        datasets=[
            PointCloudDataset(
                root=args.dataset_root,
                dataset=args.dataset,
                split='train',
                resolution=resl,
                transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min,
                                                    rotate=args.aug_rotate)
            ) for resl in args.resolutions
        ],
        split='train',
        patch_size=args.patch_size,
        num_patches=args.patches_per_shape_per_epoch,
        patch_ratio=args.patch_ratio,
        on_the_fly=True  # Currently, we only support on_the_fly=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=int(args.num_workers),
        pin_memory=True,
        shuffle=True
    )

    num_batch = len(train_dataloader)
    for batch_ind, train_batch in enumerate(train_dataloader):
        denoisenet.train()
        optimizer.zero_grad()
        pcl_noisy = train_batch['pcl_noisy'].float().cuda(non_blocking=True)
        pcl_clean = train_batch['pcl_clean'].float().cuda(non_blocking=True)
        pcl_seeds = train_batch['seed_pnts'].float().cuda(non_blocking=True)
        pcl_std = train_batch['pcl_std'].float().cuda(non_blocking=True)

        loss = denoisenet(pcl_noisy, pcl_clean, pcl_seeds, pcl_std)

        ## 反向传播更新参数
        loss.backward()
        optimizer.step()

        ## 记录训练过程
        print('[%d: %d/%d] train loss: %f\n' % (0, batch_ind, num_batch, loss.item()))
    # train_writer.add_scalar('loss', loss.data.item(), epoch * num_batch + batch_ind)

