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
from myUtils.sampling import farthest_point_sampling

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
    def __init__(self, num_modules=4, noise_decay=4):
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

            pred_disp= self.feature_nets[i](pcl_input.transpose(2, 1).contiguous())

            if self.noise_decay != 1:
                prev_std = curr_std
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
            )   # (B, N, K, 3)
            clean_nbs = clean_pts.view(B, N_noisy, d)  # (B, N, 3)

            clean_nbs = clean_nbs - pcl_input
            dist = ((pred_disp - clean_nbs)**2).sum(dim=-1) # (B, N)
            losses[i] = (seed_weights * dist).sum(dim=-1).mean(dim=-1)
        return losses.sum()  # , target, scores, noise_vecs

    def curr_iter_add_noise(self, pcl_clean, noise_std):
        new_pcl_clean = pcl_clean + torch.randn_like(pcl_clean) * noise_std.unsqueeze(1).unsqueeze(2)
        return new_pcl_clean.float()

    def patch_based_denoise(self, pcl_noisy, patch_size=1000, seed_k=5, seed_k_alpha=10, num_modules_to_use=None):
        """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
        assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.size() # 对于Icosahedron这个点云来说，N=50000
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3) # 增加一个'批'的维度

        ## 创建patch
        # 首先通过最远点采样（farthest point sampling）和k最近邻（k-nearest neighbors）来创建点云的patch。
        # 每个patch包含一个种子点及其最近的patch_size-1个点。
        num_patches = int(seed_k * N / patch_size) # 对于Icosahedron这个点云来说,num_patches=300
        # utils.farthest_point_sampling(原点云，采样点数)返回采样点组成的tensor，以及其在原点云中的索引
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches) # 对于Icosahedron这个点云来说，seed_pnts.shape=(1, 300, 3)
        # dists, idx, nn = knn_points(p1,p2,k,return_nn),return_nn被设置为True,则返回p1中每个点在p2中K个最近邻
        # dists：三维tensor，包含p1到p2中k个最近邻的距离。本例中dists.shape(1, 300, 1000)
        # idx：三维tensor，包含p1在p2中k个最近邻的下标。本例中idx.shape(1, 300, 1000)
        # nn：四维tensor，（点云数，一个点云中的点数P1，K，每个点的维数D）。本例中nn.shape(1, 300, 1000, 3)
        patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(p1=seed_pnts, p2=pcl_noisy, K=patch_size, return_nn=True)

        ## patch预处理
        # 为了使patch的处理更加方便，将每个patch中的点的坐标转换为相对于其种子点的坐标。
        patches = patches[0]    # (N, K, 3)，对于Icosahedron这个点云来说patches.shape=(300, 1000, 3)

        # Patch stitching preliminaries
        # seed_pnts.shape: (1, 300, 3)
        # seed_pnts.squeeze().shape: (300, 3)
        # seed_pnts.squeeze().unsqueeze(1).shape: (300, 1, 3)
        # seed_pnts_1.shape(300, 1000, 3)
        # 因为pytorch3d.ops.knn_points对300个采样点每个都找了1000个最邻近点，所以要把原来的300个采样点每个复制1000次
        seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)

        patches = patches - seed_pnts_1
        # patch_dists[0]取了第一个点云出来（虽然只有一个点云），2维tensor，每一行包含一个采样点到1000个最近邻的距离
        # point_idxs_in_main_pcd[0]取了第一个点云出来（虽然只有一个点云），2维tensor，每一行包含一个采样点的1000个最近邻在原点云中的索引
        patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]


        ## 计算距离比例
        # 根据之前计算的每个点到其种子点的距离，并将这些距离转换为相对于每个pathc中最远点的距离的比例。
        # patch_dists.shape(300,1000)
        # patch_dists[:, -1].shape(300)
        # patch_dists[:, -1].unsqueeze(1).shape = (300, 1)
        # .repeat(1, patch_size):将最后一个维度复制1000次，得到的shape(300,1000)
        # patch_dists.shape(300, 1000)
        patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)


        all_dists = torch.ones(num_patches, N) / 0 # all_dists.shape=(300, 50000)
        all_dists = all_dists.cuda()
        all_dists = list(all_dists)
        # patch_dists.shape(300, 1000)
        # point_idxs_in_main_pcd.shape(300, 1000)
        # all_dists.shape=(300, 50000)
        patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)
        for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists):
            # all_dist.shape(50000)
            # patch_id.shape(1000)
            # patch_dist.shape(1000)
            all_dist[patch_id] = patch_dist
        all_dists = torch.stack(all_dists,dim=0) # all_dists本是一维tensor的列表，使用torch.stack(,dim=0),将tensor列表转成2维tonsor
        # all_dists.shape: torch.Size([300, 50000])


        ## 计算权重
        # 使用这些距离来计算每个点的权重，权重用于后续的去噪步骤。权重是通过对距离应用指数函数并取最大值来计算的。
        weights = torch.exp(-1 * all_dists)
        # weights.shape(300,50000)
        # torch.max(weights, dim=0)，对于50000列中每一列在300个行中取最大值，shape(50000)
        # best_weights_idx每一列中最大值所在的行索引，即这个点的最大值是哪个patch的
        best_weights, best_weights_idx = torch.max(weights, dim=0) # shape(50000)


        ## 去噪
        patches_denoised = []
        # Denoising
        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size)) # patch_step = 5
        # print("patch_step:", patch_step)
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            # print("Processed {:d}/{:d} patches.".format(i, num_patches))
            # curr_patches.shape(patch_step, K近邻的K=1000, 3).
            curr_patches = patches[i:i+patch_step]
            try:
                if num_modules_to_use is None:
                    patches_denoised_temp, _ = self.denoise_langevin_dynamics(curr_patches, num_modules_to_use=self.num_modules)
                else:
                    patches_denoised_temp, _ = self.denoise_langevin_dynamics(curr_patches, num_modules_to_use=num_modules_to_use)

            except Exception as e:
                print("="*100)
                print(e)
                print("="*100)
                print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.")
                print("Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
                print("="*100)
                return
            patches_denoised.append(patches_denoised_temp)
            i += patch_step
        # patches_denoised是tensor列表，其每个元素tesnor的shape(5, 1000, 3)
        patches_denoised = torch.cat(patches_denoised, dim=0)
        # patches_denoised是3维tensor，shape(300, 1000, 3)
        patches_denoised = patches_denoised + seed_pnts_1
        ## Patch stitching patch拼接
        pcl_denoised = [patches_denoised[patch][point_idxs_in_main_pcd[patch] == pidx_in_main_pcd] for pidx_in_main_pcd, patch in enumerate(best_weights_idx)]
        # pcl_denoised是1维tensor列表，长度50000，每个元素是表示一个3维点的一维tensor
        pcl_denoised = torch.cat(pcl_denoised, dim=0)
        # pcl_denoised.shape[50000, 3]
        return pcl_denoised

    def denoise_langevin_dynamics(self, pcl_noisy, num_modules_to_use):
        B, N, d = pcl_noisy.size()
        pred_disps = []
        pcl_inputs = []

        with torch.no_grad():
            # print("[INFO]: Denoising up to {} iterations".format(num_modules_to_use))
            # print("self.feature:\n", self.feature_nets)
            for i in range(num_modules_to_use):
                self.feature_nets[i].eval()

                if i == 0:
                    pcl_inputs.append(pcl_noisy)
                else:
                    pcl_inputs.append(pcl_inputs[i-1] + pred_disps[i-1])
                pred_points= self.feature_nets[i](pcl_inputs[i].transpose(2, 1).contiguous())  # (B, N, F)

                pred_disps.append(pred_points)

        return pcl_inputs[-1] + pred_disps[-1], None



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

