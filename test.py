import os
import torch
import numpy as np

import argparse
from multiprocessing import Pool
from myUtils.transforms import *
from myUtils.misc import *
from denoise1 import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--noise_lvls', type=str_list, default=['0.01'])  # Set your test noise level
parser.add_argument('--resolutions', type=str_list, default=['50000_poisson'])  # Set your test resolution
parser.add_argument('--input_root', type=str, default='./data/examples')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--output_root', type=str, default='./data/results')
parser.add_argument('--eval_dir', type=str, default='./pretrained', help='')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--niters', type=int, default=1)
parser.add_argument('--patch_stitching', type=bool, default=True, help='Use patch stitching or not?')
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--seed_k', type=int, default=6)  # 6 for Kinect, 6 for small PCL, 6 for RueMadame PCL
parser.add_argument('--seed_k_alpha', type=int, default=10)  # 2 for Kinect, 10 for small PCL, 20 for RueMadame PCL
parser.add_argument('--num_modules_to_use', type=int, default=None)



args = parser.parse_args()


def input_iter(input_dir):
    # os.listdir(input_dir)把input_dir文件夹下的文件或文件夹名以字符串列表的形式返回['camel.xyz','casting.xyz',...]
    for fn in sorted(os.listdir(input_dir)):
        # fn是文件名，如famel.xyz
        if fn[-3:] != 'xyz':
            # 如果fn所代表的文件不是xzy文件，就跳过
            continue
        # os.path.join(input_dir, fn)把文件夹的相对路径和文件名拼接起来，形成文件的相对路径
        # np.loadtxt读入.xyz文件，返回2维的numpy.ndarray
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        # pcl_noisy二维，一行一个三维点
        # NormalizeUnitSphere.normalize对点云进行归一化;
        # center，2维tensor，shape(1,3)，是点云的原中心,[[(Xmax+Xmin)/2, (Ymax+Ymin)/2, (Zmax+Zmin)/2]]
        # scale，2维tensor，shape(1,1)，将点云的中心移到原点后，找点云距离中心的最大欧氏距离
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)

        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }


def main(noise):
    for resolution in args.resolutions:
        input_dir = os.path.join(args.input_root, '%s_%s_%s' % (args.dataset, resolution, noise))
        save_title = '{dataset}_Ours{modeltag}_{tag}_{res}_{noise}'.format_map({
            'dataset': args.dataset,
            'modeltag': '' if args.niters == 1 else '%dx' % args.niters,
            'tag': args.tag,
            'res': resolution,
            'noise': noise
        })

        output_dir = os.path.join(args.output_root, save_title)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Output point clouds

        ## 3.实例化模型,加载模型
        pointfilter_eval = DenoiseNet()
        model_filename = os.path.join(args.eval_dir, 'model_full_ae_6.pth')
        checkpoint = torch.load(model_filename)
        pointfilter_eval.load_state_dict(checkpoint['state_dict'])
        pointfilter_eval.cuda()

        # Denoise
        for data in input_iter(input_dir):
            pcl_noisy = data['pcl_noisy'].to(args.device)  # 数据移到cuda上
            with torch.no_grad():
                pointfilter_eval.eval()
                pcl_next = pcl_noisy
                for _ in range(args.niters):
                    if args.patch_stitching:
                        pcl_next = pointfilter_eval.patch_based_denoise(pcl_noisy=pcl_next,
                                                             patch_size=args.patch_size,
                                                             seed_k=args.seed_k,
                                                             seed_k_alpha=args.seed_k_alpha,
                                                             num_modules_to_use=args.num_modules_to_use)
                    elif not args.patch_stitching:
                        print("代码还没写完")

                pcl_denoised = pcl_next.cpu()
                # Denormalize
                pcl_denoised = pcl_denoised * data['scale'] + data['center']

            save_path = os.path.join(output_dir, data['name'] + '.xyz')
            np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')
            break



if __name__ == '__main__':
    seed_all(args.seed)
    # args.noise_lvls = ['0.01']
    with Pool(len(args.noise_lvls)) as p:
        p.map(main, args.noise_lvls)