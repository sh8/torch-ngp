import os
import glob
import numpy as np

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays_from_grids


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],
                        dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=0.5)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b],
                         [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size,
               device,
               radius=1,
               theta_range=[np.pi / 3, 2 * np.pi / 3],
               phi_range=[0, 2 * np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (
        theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(
        size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ],
                          dim=-1)  # [B, 3]

    # lookat
    forward_vector = -normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(
        size, 1)  # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float,
                      device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector),
                                   dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:

    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.mode = opt.mode  # colmap, blender, llff

        self.num_of_views = 5
        self.radius = 1.0
        self.H = 256
        self.W = 256
        self.grid_size = 32

        obj_id = '03001627'  # chair
        obj_paths = glob.glob(
            os.path.join(self.root_path, obj_id,
                         '**/models/model_normalized.obj'))
        self.obj_paths = obj_paths
        self.fx = 1.5  # NDC
        self.fy = 1.5  # NDC
        self.intrinsics = np.array([
            self.fx * self.W / 2, self.fx * self.H / 2, self.W / 2, self.H // 2
        ])

    def collate(self, index):
        poses = rand_poses(self.num_of_views, self.device, radius=self.radius)
        rays = get_rays_from_grids(poses, self.intrinsics, self.H, self.W,
                                   self.grid_size)
        # visualize_poses(poses.cpu().numpy(), size=0.1)

        obj_path = self.obj_paths[index[0]]
        # print(vertices.shape)
        # colors, _ = load_mtl(obj_path.replace("obj", "mtl"))
        # print(colors)
        results = {
            'H': self.H,
            'W': self.W,
            'fx': self.fx,
            'fy': self.fy,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'obj_path': obj_path,
            'poses': poses,
        }
        return results

    def dataloader(self):
        size = len(self.obj_paths)
        loader = DataLoader(list(range(size)),
                            batch_size=1,
                            collate_fn=self.collate,
                            shuffle=True,
                            num_workers=0)
        return loader
