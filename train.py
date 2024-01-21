import argparse
import os
import torch.utils.data
import torch.optim as optim
import numpy as np

from denoise1 import DenoiseNet
from myUtils.adjustLR import *
from myUtils.myLoss import *
from myUtils.misc import str_list
from myUtils.pcl import PointCloudDataset
from myUtils.patch import PairedPatchDataset
from myUtils.transforms import standard_train_transforms


def train(args):


    ## 1.创建日志



    ## 2.构建保存网络的目录
    # args.network_model_dir ./Summary/Train
    if not os.path.exists(args.network_model_dir):
        os.makedirs(args.network_model_dir)


    ## 3.创建种子
    # np.random.seed(args.manualSeed)
    # torch.manual_seed(args.manualSeed)


    ## 4.实例化网络
    denoisenet = DenoiseNet(args.num_modules, args.noise_decay).cuda()


    # 5.实例化优化器
    # args.lr 1e-2
    # args.momentum 0.9
    optimizer = optim.SGD(
        denoisenet.parameters(),
        lr=args.lr,
        momentum=args.momentum)


    ## 5.2加载pth
    # resume = './Summary/Train/model_full_ae_8.pth'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            denoisenet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    ## 6.构建数据集
    # 点云数据集
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
    # 一批对齐后且个数为patch_size=1000的噪声patch，三维tensor，shape(B32, 1000, 3)
    # 一批对齐后且个数为patch_size*ratio=1200的干净patch，三维tensor，shape(B32, 1200, 3)
    # 一批对齐后且个数为patch_size*ratio=1200的干净patch法线，三维tensor，shape(B32, 1200, 3)
    # 一批upport_radius支持半径是二维tensor，shape[B32, 1]

    num_batch = len(train_dataloader) # 132个点云，每个点云取8000个patch，批大小是32，所以132 * 8000 / 32 = 33000
    # opt.start_epoch=0
    # opt.nepoch=50
    for epoch in range(args.start_epoch, args.nepoch):
        # 调整学习率
        adjust_learning_rate(optimizer, epoch, args)
        print('lr is %.10f' % (optimizer.param_groups[0]['lr']))
        for batch_ind, train_batch in enumerate(train_dataloader):
            denoisenet.train()
            optimizer.zero_grad()
            pcl_noisy = train_batch['pcl_noisy'].float().cuda(non_blocking=True)
            pcl_clean = train_batch['pcl_clean'].float().cuda(non_blocking=True)
            pcl_seeds = train_batch['seed_pnts'].float().cuda(non_blocking=True)
            pcl_std = train_batch['pcl_std'].float().cuda(non_blocking=True)

            ## 计算损失
            #要输入原来的噪声点noise_patch，以及位移pred_dis，以及干净点gt_patch
            loss = denoisenet(pcl_noisy, pcl_clean, pcl_seeds, pcl_std)

            ## 反向传播更新参数
            loss.backward()
            optimizer.step()

            ## 记录训练过程
            print('[%d: %d/%d] train loss: %f\n' % (epoch, batch_ind, num_batch, loss.item()))
        # train_writer.add_scalar('loss', loss.data.item(), epoch * num_batch + batch_ind)

        ## 保存模型
        checpoint_state = {
            'epoch': epoch + 1,
            'state_dict': denoisenet.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if epoch == (args.nepoch - 1):
            torch.save(checpoint_state, '%s/model_full_ae.pth' % args.network_model_dir)

        if epoch % args.model_interval == 0:
            torch.save(checpoint_state, '%s/model_full_ae_%d.pth' % (args.network_model_dir, epoch))




if __name__ == "__main__":
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
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--resume', type=str, default='', help='refine model at this path')

    ## train
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--model_interval', type=int, default=1)


    # Ablation parameters
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--num_modules', type=int, default=4)
    parser.add_argument('--noise_decay', type=int, default=4)  # Noise decay is set to 16/T where T=num_modules or set to 1 for no decay

    args = parser.parse_args()
    args.resume = './Summary/Train/model_full_ae_6.pth'
    args.nepoch = 50
    print(args)
    train(args)