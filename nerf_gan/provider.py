import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays


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
    sphere = trimesh.creation.icosphere(radius=1)
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
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.

        self.training = self.type in ['train', 'all']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.partition_ids = []

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'),
                      'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(
                    os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # only load one specified split
            else:
                with open(
                        os.path.join(self.root_path,
                                     f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)
        elif self.mode == 'shapenet':
            obj_id = '03001627'  # chair
            transform_paths = glob.glob(
                os.path.join(self.root_path, obj_id, '**/cameras.npz'))

            if 'ShapeNet' in self.root_path:
                transform = {'h': 256, 'w': 256}
            else:
                transform = {'h': 64, 'w': 64}

            # Load intrinsics
            transform_path = transform_paths[0]
            poses = np.load(transform_path)
            K = poses['camera_mat_0']
            transform['fl_x'] = K[0, 0] * transform['w'] // 2
            transform['fl_y'] = K[1, 1] * transform['h'] // 2
            transform['cx'] = transform['w'] // 2
            transform['cy'] = transform['h'] // 2

            total_num_imgs = 0
            for transform_path in transform_paths[:10]:
                poses = np.load(transform_path)
                num_imgs = len(poses.files) // 4
                self.partition_ids.append(
                    (total_num_imgs, total_num_imgs + num_imgs))
                total_num_imgs += num_imgs

                frames = []
                for i in range(num_imgs):
                    file_path = transform_path.replace('cameras.npz',
                                                       f'image/{i:04d}.png')
                    file_path = file_path.replace(self.root_path, '', 1)[1:]
                    R = np.eye(4)
                    R[1, 1] = -1.0
                    R[2, 2] = -1.0
                    Rt = poses[f'world_mat_inv_{i}']
                    transform_mat = Rt @ R
                    frame = {}
                    frame['file_path'] = file_path
                    frame['transform_matrix'] = transform_mat.tolist()
                    frames.append(frame)
                transform['frames'].extend(frames)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])

        self.poses = []
        self.images = None
        for f in tqdm.tqdm(frames, desc=f'Loading {type} data:'):
            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale)

            self.poses.append(pose)

        self.poses = torch.from_numpy(np.stack(self.poses,
                                               axis=0))  # [N, 4, 4]
        self.frames = frames

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        # print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x']
                    if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y']
                    if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)
                             ) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)
                             ) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError(
                'Failed to load focal length, please check the transforms.json!'
            )

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H /
                                                                      2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W /
                                                                      2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):
        partition_id = self.partition_ids[index[0]]
        start_ind = partition_id[0]
        end_ind = partition_id[1]
        frames = self.frames[start_ind:end_ind]
        poses = self.poses[start_ind:end_ind]

        B = len(frames)

        if self.images is None:
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue

                if self.mode == 'shapenet':
                    image = cv2.imread(f_path)  # [H, W, 3]
                else:
                    image = cv2.imread(
                        f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                image = cv2.resize(image, (self.W, self.H),
                                   interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255  # [H, W, 3/4]
                self.images.append(image)
            self.images = torch.from_numpy(np.stack(self.images, axis=0))

        poses = poses.to(self.device)  # [B, 4, 4]

        rays = get_rays(poses, self.intrinsics, self.H, self.W, 128)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'poses': poses,
        }

        images = self.images.to(self.device)  # [B, H, W, 3/4]
        if self.training:
            C = images.shape[-1]
            images = torch.gather(images.view(B, -1, C), 1,
                                  torch.stack(C * [rays['inds']],
                                              -1))  # [B, N, 3/4]
        results['images'] = images

        return results

    def dataloader(self):
        size = len(self.partition_ids)
        loader = DataLoader(list(range(size)),
                            batch_size=1,
                            collate_fn=self.collate,
                            shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access poses in trainer.
        return loader
