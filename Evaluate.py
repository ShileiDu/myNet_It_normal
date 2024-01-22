import os
import torch
from tqdm.auto import tqdm
import numpy as np
import pytorch3d.loss
import point_cloud_utils as pcu
from scipy.spatial.transform import Rotation
import pandas as pd

from myUtils.distance import *

def load_xyz(xyz_dir):
    """
    将xyz_dir文件夹下的所有的.xyz文件读入，并转换成torch.FloatTensor形式。并显示读入的进度条
    Args:
        xyz_dir:包含点云的文件夹路径

    Returns:字典，key是点云的名字，value是2维形式的tensor

    """
    # xyz_dir=./data/results\PUNet_Ours__50000_poisson_0.01
    all_pcls = {}
    dir_list = sorted(os.listdir(xyz_dir))
    dir_list.sort()
    for fn in tqdm(dir_list, desc='Loading PCLs'):
        if fn[-3:] != 'xyz':
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls

def load_off(off_dir):
    """
    将off_dir文件夹下的.off文件读入，并转换成tensor格式，返回一个字典
    Args:
        off_dir:包含.off文件的文件夹.

    Returns:字典。key是文件名，value还是也一个字典；value.key是该off的顶点tensor，value.value是该off的面tensor

    """
    # off_dir=./data/PUNet/meshes/test
    # .off 是一种3D文本格式，通过定义点、线、面的方式来描述3D物体
    all_meshes = {}
    dir_list = sorted(os.listdir(off_dir))
    dir_list.sort()
    for fn in tqdm(dir_list, desc='Loading meshes'):
        if fn[-3:] != 'off':
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        verts, faces = pcu.load_mesh_vf(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {'verts': verts, 'faces': faces}
    return all_meshes



class Evaluator(object):

    def __init__(self, output_pcl_dir, dataset_root, dataset, summary_dir, experiment_name, device='cuda', res_gts='8192_poisson'):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir # ./data/results\PUNet_Ours__50000_poisson_0.01
        self.dataset_root = dataset_root # ./data
        self.dataset = dataset # PUNet
        self.summary_dir = summary_dir # ./data/results
        self.experiment_name = experiment_name
        self.gts_pcl_dir = os.path.join(dataset_root, dataset, 'pointclouds', 'test', res_gts) # ./data\PUNet\pointclouds\test\50000_poisson
        self.gts_mesh_dir = os.path.join(dataset_root, dataset, 'meshes', 'test') # ./data/PUNet/meshes/test
        self.res_gts = res_gts
        self.device = device
        # self.logger = logger
        self.load_data()

    def load_data(self):
        # 将self.output_pcl_dir文件夹下的所有预测后的.xyz文件读入，并转换成torch.FloatTensor形式。并显示读入的进度条
        self.pcls_pred = load_xyz(self.output_pcl_dir)
        # print("self.pcls_pred", self.pcls_pred)
        # 将.\data\PUNet\pointclouds\test\50000_poisson文件夹下所有测试用的.xyz文件读入，并转换成torch.FloatTensor形式。并显示读入的进度条
        self.pcls_gt = load_xyz(self.gts_pcl_dir)
        # print("self.pcls_gt", self.pcls_gt)
        # 将./data/PUNet/meshes/test文件夹下所有测试用的.off文件读入，并转换成tensor格式。并显示读入的进度条
        self.meshes = load_off(self.gts_mesh_dir)
        # print("self.meshes", self.meshes)
        # 预测后点云的名字列表
        self.pcls_pred_name = list(self.pcls_pred.keys())
        # print("self.pcls_pred_name", self.pcls_pred_name)
        # 真实点云的名字列表
        self.pcls_gt_name = list(self.pcls_gt.keys())
        # self.pcls_pred[Icosahedron]        torch.Size([50000, 3])
        # self.pcls_gt[Icosahedron]          torch.Size([50000, 3])
        # self.meshes[Icosahedron]['verts']  torch.Size([2562, 3])
        # self.meshes[Icosahedron]['faces']  torch.Size([5120, 3])

    def run(self):
        # pcls_pred是一个字典，key是点云的名字，value是表示预测的点云中点的一个2维tensor
        # pcls_gt是一个字典，key是点云的名字，value是表示真实点云中点的一个2维tensor
        # self.pcls_pred_name预测后点云的名字列表
        # self.pcls_gt_name真实点云的名字列表
        pcls_pred, pcls_gt, pcls_pred_name, pcls_gt_name = self.pcls_pred, self.pcls_gt, self.pcls_pred_name, self.pcls_gt_name
        results = {}
        results_cd = {}
        results_p2f = {}
        for pred_name, gt_name in tqdm(zip(pcls_pred_name, pcls_gt_name), desc='Evaluate'):
            # print("pcls_pred[pred_name].shape", pcls_pred[pred_name].shape)
            # print("pcls_pred[pred_name][:,:3].shape", pcls_pred[pred_name][:,:3].shape)
            # print("pcls_pred[pred_name][:,:3].unsqueeze(0).shape", pcls_pred[pred_name][:,:3].unsqueeze(0).shape)
            # pcls_pred[pred_name].shape(50000, 3)
            # pcls_pred[pred_name][:,:3].shape(50000,3)
            # pcls_pred[pred_name][:,:3].unsqueeze(0).shape(1,50000,3)
            pcl_pred = pcls_pred[pred_name][:,:3].unsqueeze(0).to(self.device) # shape(1,50000,3)
            if gt_name not in pcls_gt:
                # self.logger.warning('Shape `%s` not found, ignored.' % pcls_gt_name)
                print('Shape `%s` not found, ignored.' % pcls_gt_name)

                continue
            pcl_gt = pcls_gt[gt_name].unsqueeze(0).to(self.device)# pcl_gt.shape(1,50000,3)

            verts = self.meshes[gt_name]['verts'].to(self.device)# shape(顶点数,3)
            faces = self.meshes[gt_name]['faces'].to(self.device) # shape(面数,3)

            # chamfer_distance()返回一个2元组，第一项表示chamfer distance的tensor
            cd = pytorch3d.loss.chamfer_distance(pcl_pred, pcl_gt)[0].item()
            # print("cd", cd)
            cd_sph = chamfer_distance_unit_sphere(pcl_pred, pcl_gt)[0].item()
            # print("cd_sph", cd_sph)
            hd_sph = hausdorff_distance_unit_sphere(pcl_pred, pcl_gt)[0].item()

            # p2f = point_to_mesh_distance_single_unit_sphere(
            #     pcl=pcl_pred[0],
            #     verts=verts,
            #     faces=faces
            # ).sqrt().mean().item()
            if 'blensor' in self.experiment_name:
                print("blensor", True)
                rotmat = torch.FloatTensor(Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()).to(pcl_pred[0])
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_pred[0].matmul(rotmat.t()),
                    verts=verts,
                    faces=faces
                ).item()
            else:
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_pred[0],
                    verts=verts,
                    faces=faces
                ).item()

            cd *= 10000
            cd_sph *= 10000
            p2f *= 10000

            results[gt_name] = {
                'cd': cd,
                'cd_sph': cd_sph,
                'p2f': p2f,
                # 'hd_sph': hd_sph,
            }
        results = pd.DataFrame(results).transpose()
        results_cd = pd.DataFrame(results_cd)
        results_p2f = pd.DataFrame(results_p2f)
        res_mean = results.mean(axis=0)
        print("res_mean:", res_mean)
        # self.logger.info("\n" + results.to_string())
        print("\n" + results.to_string())

        # self.logger.info("\nMean\n" + '\n'.join([
        #     '%s\t%.12f' % (k, v) for k, v in res_mean.items()
        # ]))

        print("\nMean\n" + '\n'.join(['%s\t%.12f' % (k, v) for k, v in res_mean.items()]))