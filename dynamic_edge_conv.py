import torch
from torch.nn import Sequential as Seq, Linear, BatchNorm1d as BN, ReLU
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        """
        MessagePassing初始化:
        aggr:邻域聚合方式，默认add，还可以是mean, max。得到每个邻居传递给中心节点的消息后，我们需要用一种可微且置换不变（ permutation invariant）的函数来聚合邻域消息。
        flow:消息传递方向，默认从source_to_target，也可以设置为target_to_source，不过source_to_target是最通常的传递机制，也就是从节点j传递消息到节点i。
        node_dim:定义沿着哪个维度进行消息传递，默认-2，因为-1是特征维度。
        """
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       BN(out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels),
                       BN(out_channels),
                       ReLU()
                       )
        self.lin = Seq(Linear(in_channels, out_channels),
                       BN(out_channels),
                       ReLU()
                       )

    def forward(self, x, edge_index):
        # x has shape [N, in_channels] # 图就是每个点的特征。x.shape[批大小4 * patch_size1000, 初始的点维度3]
        # edge_index: has shape [2, E] # 邻接矩阵。提供了为消息如何传递提供了信息它有两种形式：Tensor和SparseTensor。Tensor形式下的
        # edge_index的shape是(2, N)。SparseTensor则可以理解为稀疏矩阵的形式存储边信息。

        # edge_index.shape(2, 批大小4 * patch_size1000 * 32)
        x_pair = (x, x)
        # print("edge_index:\n", edge_index)
        # print("edge_index.shape:", edge_index.shape)

        out_1 = self.propagate(edge_index, x=x_pair[0]) # 所有的逻辑代码都在forward()里面，当我们调用propagate()函数之后，它将会在内部调用message()和update()。
        # print("out_1.shape", out_1.shape)
        out_2 = self.lin(x_pair[1])

        return out_1 + out_2

    def message(self, x_i, x_j):
        """

        在flow='source_to_target'的设置下，计算了邻居节点j到中心节点i的消息。
        Args:
            x_i: has shape [边数E, 特征数in_channels]，每一行是edge_index中每条边终点的特征。edge_index的第二行是一个1维tensor，保存着每条边的终点索引，以该索引为下标在x中找特征，作为x_i的一行。
            x_j: has shape [边数E, 特征数in_channels]，每一行是edge_index中每条边起点的特征。edge_index的第一行是一个1维tensor，保存着每条边的起点索引，以该索引为下标在x中找特征，作为x_j的一行。

        Returns:

        """
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x, edge_index):
        return super().forward(x, edge_index)