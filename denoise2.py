#真实训练时不需要，模拟训练时需要
import argparse
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

##################################该文件需要
import torch
import torch.nn as nn
import pytorch3d.ops
from torch_geometric.utils import remove_self_loops

from dynamic_edge_conv import DynamicEdgeConv



def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(y, x, K=k+offset)
    return knn_idx[:, :, offset:]


class IterativeEncoder(nn.Module):
    def __init__(self, k=32, num_keyPoint=10, output_dim=3):
        super(IterativeEncoder, self).__init__()
        self.k = k
        self.num_keyPoint = num_keyPoint
        self.output_dim = output_dim
        ## 用于多特征提取
        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 48)

        ## 用于形状感知选择
        self.linear1 = nn.Linear(1, 64) # 用于距离信息
        self.linear2 = nn.Linear(48, 64) # 用于近邻点信息
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.LeakyReLU(negative_slope=0.2))

        ## 用于特征融合,目前只有针对点特征的，还没有法线特征的
        self.mlp1 = nn.Sequential(nn.Conv1d(in_channels=48 * num_keyPoint, out_channels=512, kernel_size=1),
                                 nn.BatchNorm1d(num_features=512),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                 nn.BatchNorm1d(num_features=512),
                                 nn.LeakyReLU(negative_slope=0.2)
                                 )
        self.mlp2 = nn.Sequential(nn.Linear(48 * num_keyPoint, 512),
                                  nn.BatchNorm1d(num_features=512),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(num_features=512),
                                  nn.LeakyReLU(negative_slope=0.2)
                                  )

        ## Decoder
        self.decoder = nn.Sequential( nn.Linear(512, 256, bias=False),
                                      nn.Linear(256, 128),
                                      nn.Linear(128, self.output_dim))


    def forward(self, x):
        ## 1.多特征提取（阶段/器）,以下只是针对点特征的提取。之后还应该把下边的步骤封装，用于法线的提取
        # 两个Dynamic Edge Conv
        self.batch_size = x.size(0)
        self.num_points = x.size(1)

        self.rows = torch.arange(0, self.num_points).unsqueeze(0).unsqueeze(2).repeat(self.batch_size, 1, self.k+1).cuda()
        # self.rows.shape=(批大小4, patch size1000, 33)

        self.rows_add = (self.num_points*torch.arange(0, self.batch_size)).unsqueeze(1).unsqueeze(2).repeat(1, self.num_points, self.k+1).cuda()
        # self.rows_add.shape=[4, 1000, 33]

        # edge_index是二维的，每一列是一条边，第一行是边的源，第二行是边的目的地。self.rows是edge_index第二行，33个0，33个1，33个2,...,33个3999
        self.rows = (self.rows + self.rows_add).view(1, -1) # shape(批大小4，patch size1000, 自己和邻居的个数33)
        # self.rows.shape=[1, 132000]

        # x.shape[4, 1000, 3]，一批有4个patch，每个patch1000个3维点
        edge_index = self.get_edge_index(x) # 将一批的数据x做成一个图，返回edge_index。虽然一个图一个的点云不同patch，但是他们之间不会有边，每个点只与自己patch中最近的32个点形成边。
        x = x.view(self.batch_size * self.num_points, -1) # 一行一个三维点。x.shape[4*1000, 3]
        # edge_index.shape(2,128000)
        x1 = self.conv1(x, edge_index) # 输出x1.shape[4000, 16]
        x1 = x1.view(self.batch_size, self.num_points, -1) # x1.shape[4, 1000, 16]

        edge_index = self.get_edge_index(x1) # shape(2, 批大小4*patch_size10000*32)
        x1 = x1.view(self.batch_size*self.num_points, -1) # shape(4000, 16)
        x2 = self.conv2(x1, edge_index) # shape(4000, 48) # shape(4000, 48)
        # x2 = x2.view(self.batch_size, self.num_points, -1) # shape(4, 1000, 48)

        ## 2.形状感知选择器。还应该结合法线的边卷积之后的结果
        # 根据每个点一开始的坐标选出近邻点，得到近邻点索引idx;根据索引idx，取出近邻点经过两次边卷积之后(即x2)的特征,每个点的32近邻点的48维特征，即knnfeature
        x = x.view(self.batch_size, self.num_points, -1)
        # x.shape[32, 1000, 3]
        dist, idx, _ = pytorch3d.ops.knn_points(x, x, K=self.k+1)
        idx += self.rows_add
        idx = idx.view(-1)
        knnfeature = x2[idx].view(self.batch_size, self.num_points, self.k+1, -1) # 近邻点特征
        knnfeature = self.linear2(knnfeature) # shape(B16, patct_size1000, KNN33, num_feature64)

        dist = dist.unsqueeze(3) # 距离信息
        dist = torch.exp(-dist)
        dist = self.linear1(dist)  # shape(B16, patct_size1000, KNN33, num_feature64)

        feature = torch.cat([knnfeature, dist], dim=3) # feature.shape(16, 1000, 33, 128)
        # 全连接层参数太大。切换维度，改成核大小为1的卷积层
        feature = feature.permute(0,3,1,2)
        feature = self.conv3(feature) # feature.shape(B8, num_feature128, patct_size10001000, KNN33)
        feature = feature.permute(0,2,3,1) # feature.shape(B8, patct_size10001000, KNN33, num_feature128)

        #.每个点挑出最大的一维特征
        feature = torch.max(feature, dim=3)[0] # 取出每个近邻点的128维特征中的最大值
        # feature.shape(8, 1000, 33)
        top_idx_inKnn = torch.argsort(feature, dim=-1, descending=True)[:, :,0:self.num_keyPoint] # shape(B8, P1000, keyPoint10) # 得到每个点的关键点在33个近邻点中的索引
        # 得到每个点的关键点在原patch中的索引
        # print("top_idx", top_idx)
        idx = idx.view(self.batch_size, self.num_points, -1) # shape(8, 1000, 33) # shape(B8, P1000, keyPoint10)
        top_idx_inPatch = torch.gather(idx, dim=-1, index=top_idx_inKnn).view(-1)
        # 取出关键点特征
        keyFeature = x2[top_idx_inPatch].view(self.batch_size, self.num_points, self.num_keyPoint, -1) # keyFeature.shape(8, 1000, 10, 48)
        # print("keyFeature1.shape", keyFeature.shape)


        ## 3.特征融合
        # 关键点特征拼接成每个点的特征, 之后可以把特征提取阶段的的输出特征扩大，将这里改成求关键点的特征的平均和最大值，然后卷积到512维。法线特征同样
        keyFeature = keyFeature.view(self.batch_size, self.num_points,-1)# keyFeature.shape(8, 1000, 480)
        keyFeature = self.mlp1(keyFeature.transpose(2, 1).contiguous()).transpose(2, 1).contiguous() # mlp的输入输出shape(B,C,N)
        # keyFeature.shape(8, 1000, 512)

        # 对法线关键点特征的处理
        # keyFeature_normal = keyFeature_normal.view(self.batch_size, self.num_points,-1)# keyFeature.shape(8, 1000, 480)
        # keyFeature_normal = self.mlp2(keyFeature_normal.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()  # mlp的输入输出shape(B,C,N)

        ## 关键点法线特征和关键点坐标特征融合，目前阶段没有


        return keyFeature


    def get_edge_index(self, x):
        """
        为一批patch中每个patch的每个点在自己的patch中找self.k个最近邻，加上自己总共self.k+1个点。得到这些点的索引，作为edge_index的第一行，边的起点
        Args:
            x:三维tensor，shape(批大小, patch size, 点的维度3)

        Returns:二维tensor，edge_index
        """
        # get_knn_idx(x, y, m), x这批点云中的每个点，在y中的m个近邻的索引
        # get_knn_idx(x, x, self.k + 1).shape = (4, 1000, 33)
        cols = get_knn_idx(x, x, self.k+1).view(self.batch_size, self.num_points, -1)
        # cols.shape=(4, 1000, 33)

        # 原来cols中的索引是相对于自己patch的，如[x, 4, 9]可能等于[y, 4, 23]，原本第x个patch中可能有下标6到下标4的边，第二个patch中也可能有下标6到下标4的边，
        # 将他们画在一个图会冲突，所以给他们每个点添加一个偏移，第一个patch的偏移为0，第二个的为1000，第三个的为2000，第四个的为3000
        # cols加上self.rows_add，使索引的起点相对于0
        cols = (cols + self.rows_add).view(1, -1) # cols.shape(1, 4 * 1000 * 33)

        # 将第一行边的起点列表cols与第二行边的终点列表self.rows拼接起来，形成edge_index
        edge_index = torch.cat([cols, self.rows], dim=0)
        # edge_index.shape(2,132000)

        # 去掉图中的自环。求knn的时候会把自己的也算进去（所以求32个knn，输入参数33），最后会导致有自己到自己的边，用remove_self_loops移除自环, 会去掉4(批大小) * 1000(patch size)条边
        edge_index, _ = remove_self_loops(edge_index.long())
        # edge_index.shape(2, 128000)

        return edge_index




class IterativeDecoder(nn.Module):
    def __init__(self):
        super(IterativeDecoder, self).__init__()

    def forward(self):
        pass


class IterationModule(nn.Module):
    def __init__(self, k=32, num_keyPoint=10, output_dim=3):
        super(IterationModule, self).__init__()
        self.k = k
        self.num_keyPoint = num_keyPoint
        self.output_dim = output_dim
        ## 用于多特征提取
        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 48)

        ## 用于形状感知选择
        self.linear1 = nn.Linear(1, 64)  # 用于距离信息
        self.linear2 = nn.Linear(48, 64)  # 用于近邻点信息
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.LeakyReLU(negative_slope=0.2))

        ## 用于特征融合,目前只有针对点特征的，还没有法线特征的
        self.mlp1 = nn.Sequential(nn.Conv1d(in_channels=48 * num_keyPoint, out_channels=512, kernel_size=1),
                                  nn.BatchNorm1d(num_features=512),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                  nn.BatchNorm1d(num_features=512),
                                  nn.LeakyReLU(negative_slope=0.2)
                                  )
        self.mlp2 = nn.Sequential(nn.Linear(48 * num_keyPoint, 512),
                                  nn.BatchNorm1d(num_features=512),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(num_features=512),
                                  nn.LeakyReLU(negative_slope=0.2)
                                  )

        ## Decoder
        # self.linear1 = nn.Linear(self.output_dim, 256, bias=False)
        # self.linear2 = nn.Linear(256, 128)
        # self.linear3 = nn.Linear(128, self.output_dim)
        self.decoder = nn.Sequential(nn.Linear(512, 256, bias=False),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.output_dim)
                                     )

    def forward(self, x):
        ### Encoder
        ## 1.多特征提取（阶段/器）,以下只是针对点特征的提取。之后还应该把下边的步骤封装，用于法线的提取
        # 两个Dynamic Edge Conv
        self.batch_size = x.size(0)
        self.num_points = x.size(1)

        self.rows = torch.arange(0, self.num_points).unsqueeze(0).unsqueeze(2).repeat(self.batch_size, 1,
                                                                                      self.k + 1).cuda()
        # self.rows.shape=(批大小4, patch size1000, 33)

        self.rows_add = (self.num_points * torch.arange(0, self.batch_size)).unsqueeze(1).unsqueeze(2).repeat(1,
                                                                                                              self.num_points,
                                                                                                              self.k + 1).cuda()
        # self.rows_add.shape=[4, 1000, 33]

        # edge_index是二维的，每一列是一条边，第一行是边的源，第二行是边的目的地。self.rows是edge_index第二行，33个0，33个1，33个2,...,33个3999
        self.rows = (self.rows + self.rows_add).view(1, -1)  # shape(批大小4，patch size1000, 自己和邻居的个数33)
        # self.rows.shape=[1, 132000]

        # x.shape[4, 1000, 3]，一批有4个patch，每个patch1000个3维点
        edge_index = self.get_edge_index(
            x)  # 将一批的数据x做成一个图，返回edge_index。虽然一个图一个的点云不同patch，但是他们之间不会有边，每个点只与自己patch中最近的32个点形成边。
        x = x.view(self.batch_size * self.num_points, -1)  # 一行一个三维点。x.shape[4*1000, 3]
        # edge_index.shape(2,128000)
        x1 = self.conv1(x, edge_index)  # 输出x1.shape[4000, 16]
        x1 = x1.view(self.batch_size, self.num_points, -1)  # x1.shape[4, 1000, 16]

        edge_index = self.get_edge_index(x1)  # shape(2, 批大小4*patch_size10000*32)
        x1 = x1.view(self.batch_size * self.num_points, -1)  # shape(4000, 16)
        x2 = self.conv2(x1, edge_index)  # shape(4000, 48) # shape(4000, 48)
        # x2 = x2.view(self.batch_size, self.num_points, -1) # shape(4, 1000, 48)

        ## 2.形状感知选择器。还应该结合法线的边卷积之后的结果
        # 根据每个点一开始的坐标选出近邻点，得到近邻点索引idx;根据索引idx，取出近邻点经过两次边卷积之后(即x2)的特征,每个点的32近邻点的48维特征，即knnfeature
        x = x.view(self.batch_size, self.num_points, -1)
        # x.shape[32, 1000, 3]
        dist, idx, _ = pytorch3d.ops.knn_points(x, x, K=self.k + 1)
        idx += self.rows_add
        idx = idx.view(-1)
        knnfeature = x2[idx].view(self.batch_size, self.num_points, self.k + 1, -1)  # 近邻点特征
        knnfeature = self.linear2(knnfeature)  # shape(B16, patct_size1000, KNN33, num_feature64)

        dist = dist.unsqueeze(3)  # 距离信息
        dist = torch.exp(-dist)
        dist = self.linear1(dist)  # shape(B16, patct_size1000, KNN33, num_feature64)

        feature = torch.cat([knnfeature, dist], dim=3)  # feature.shape(16, 1000, 33, 128)
        # 全连接层参数太大。切换维度，改成核大小为1的卷积层
        feature = feature.permute(0, 3, 1, 2)
        feature = self.conv3(feature)  # feature.shape(B8, num_feature128, patct_size10001000, KNN33)
        feature = feature.permute(0, 2, 3, 1)  # feature.shape(B8, patct_size10001000, KNN33, num_feature128)

        # .每个点挑出最大的一维特征
        feature = torch.max(feature, dim=3)[0]  # 取出每个近邻点的128维特征中的最大值
        # feature.shape(8, 1000, 33)
        top_idx_inKnn = torch.argsort(feature, dim=-1, descending=True)[:, :,
                        0:self.num_keyPoint]  # shape(B8, P1000, keyPoint10) # 得到每个点的关键点在33个近邻点中的索引
        # 得到每个点的关键点在原patch中的索引
        # print("top_idx", top_idx)
        idx = idx.view(self.batch_size, self.num_points, -1)  # shape(8, 1000, 33) # shape(B8, P1000, keyPoint10)
        top_idx_inPatch = torch.gather(idx, dim=-1, index=top_idx_inKnn).view(-1)
        # 取出关键点特征
        keyFeature = x2[top_idx_inPatch].view(self.batch_size, self.num_points, self.num_keyPoint,
                                              -1)  # keyFeature.shape(8, 1000, 10, 48)
        # print("keyFeature1.shape", keyFeature.shape)

        ## 3.特征融合
        # 关键点特征拼接成每个点的特征, 之后可以把特征提取阶段的的输出特征扩大，将这里改成求关键点的特征的平均和最大值，然后卷积到512维。法线特征同样
        keyFeature = keyFeature.view(self.batch_size, self.num_points, -1)  # keyFeature.shape(8, 1000, 480)
        keyFeature = self.mlp1(keyFeature.transpose(2, 1).contiguous()).transpose(2,
                                                                                  1).contiguous()  # mlp的输入输出shape(B,C,N)
        # keyFeature.shape(8, 1000, 512)

        # 对法线关键点特征的处理
        # keyFeature_normal = keyFeature_normal.view(self.batch_size, self.num_points,-1)# keyFeature.shape(8, 1000, 480)
        # keyFeature_normal = self.mlp2(keyFeature_normal.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()  # mlp的输入输出shape(B,C,N)

        ## 关键点法线特征和关键点坐标特征融合，目前阶段没有

        ### Decoder
        keyFeature = self.decoder(keyFeature)
        return torch.tanh(keyFeature)


    def get_edge_index(self, x):
        """
        为一批patch中每个patch的每个点在自己的patch中找self.k个最近邻，加上自己总共self.k+1个点。得到这些点的索引，作为edge_index的第一行，边的起点
        Args:
            x:三维tensor，shape(批大小, patch size, 点的维度3)

        Returns:二维tensor，edge_index
        """
        # get_knn_idx(x, y, m), x这批点云中的每个点，在y中的m个近邻的索引
        # get_knn_idx(x, x, self.k + 1).shape = (4, 1000, 33)
        cols = get_knn_idx(x, x, self.k+1).view(self.batch_size, self.num_points, -1)
        # cols.shape=(4, 1000, 33)

        # 原来cols中的索引是相对于自己patch的，如[x, 4, 9]可能等于[y, 4, 23]，原本第x个patch中可能有下标6到下标4的边，第二个patch中也可能有下标6到下标4的边，
        # 将他们画在一个图会冲突，所以给他们每个点添加一个偏移，第一个patch的偏移为0，第二个的为1000，第三个的为2000，第四个的为3000
        # cols加上self.rows_add，使索引的起点相对于0
        cols = (cols + self.rows_add).view(1, -1) # cols.shape(1, 4 * 1000 * 33)

        # 将第一行边的起点列表cols与第二行边的终点列表self.rows拼接起来，形成edge_index
        edge_index = torch.cat([cols, self.rows], dim=0)
        # edge_index.shape(2,132000)

        # 去掉图中的自环。求knn的时候会把自己的也算进去（所以求32个knn，输入参数33），最后会导致有自己到自己的边，用remove_self_loops移除自环, 会去掉4(批大小) * 1000(patch size)条边
        edge_index, _ = remove_self_loops(edge_index.long())
        # edge_index.shape(2, 128000)

        return edge_index


class DenoiseNet(nn.Module):
    def __init__(self, num_modules=4, noise_decay=4):
        super(DenoiseNet, self).__init__()
        self.num_modules = num_modules
        self.noise_decay = noise_decay
        self.feature_nets = nn.ModuleList()
        for i in range(self.num_modules):
            self.feature_nets.append(IterationModule().cuda())

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

            pred_disp= self.feature_nets[i](pcl_input)

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
                print(i)
                self.feature_nets[i].eval()

                if i == 0:
                    pcl_inputs.append(pcl_noisy)
                else:
                    pcl_inputs.append(pcl_inputs[i-1] + pred_disps[i-1])
                pred_points= self.feature_nets[i](pcl_inputs[i])  # (B, N, F)

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
    parser.add_argument('--train_batch_size', type=int, default=4)
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
        # loss = (pcl_noisy, pcl_clean, pcl_seeds, pcl_std)

        ## 反向传播更新参数
        loss.backward()
        optimizer.step()

        ## 记录训练过程
        print('[%d: %d/%d] train loss: %f\n' % (0, batch_ind, num_batch, loss.item()))
    # train_writer.add_scalar('loss', loss.data.item(), epoch * num_batch + batch_ind)
