import os
import glob
import math
import tqdm
import random
import tensorboardX

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import autograd

import trimesh
import mcubes
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    BlendParams,
    PointLights,
    SoftPhongShader,
)
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        inds = torch.randint(0, H * W, size=[N],
                             device=device)  # may duplicate
        inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
        results['inds'] = None

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


@torch.cuda.amp.autocast(enabled=False)
def get_rays_from_grids(poses,
                        intrinsics,
                        H,
                        W,
                        grid_size,
                        iterations,
                        scale_anneal=0.025,
                        min_scale=0.25,
                        max_scale=1.0):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, grid_size: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''
    B = poses.shape[0]
    # grid_x = torch.randint(0, W // grid_size, size=[1])[0] * grid_size
    # grid_y = torch.randint(0, H // grid_size, size=[1])[0] * grid_size

    device = poses.device
    # B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    if scale_anneal > 0:
        k_iter = iterations // 1000 * 3
        min_scale = max(min_scale,
                        max_scale * math.exp(-k_iter * scale_anneal))
        min_scale = min(0.9, min_scale)
    else:
        min_scale = min_scale

    i, j = torch.meshgrid(torch.linspace(0, H - 1, grid_size, device=device),
                          torch.linspace(0, W - 1, grid_size, device=device),
                          indexing='ij')

    scale = torch.Tensor(1).uniform_(min_scale, max_scale)
    i = i * scale.to(device)
    j = j * scale.to(device)

    max_offset = 1 - scale.item()
    h_offset = torch.Tensor(1).uniform_(0, max_offset * H)
    w_offset = torch.Tensor(1).uniform_(0, max_offset * W)
    i = i + h_offset.to(device)
    j = j + w_offset.to(device)

    # implement Grid
    i = i.reshape([1, -1]).repeat([B, 1]).long()
    j = j.reshape([1, -1]).repeat([B, 1]).long()
    inds = j + i * W

    results = {}
    results['inds'] = inds

    zs = torch.ones_like(i)
    ys = (i - cy) / fy * zs
    xs = (j - cx) / fx * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (
            x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055)**2.4)


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([
                        xx.reshape(-1, 1),
                        yy.reshape(-1, 1),
                        zz.reshape(-1, 1)
                    ],
                                    dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(
                        len(xs), len(ys),
                        len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S:xi * S + len(xs), yi * S:yi * S + len(ys),
                      zi * S:zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (
        b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:

    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean(np.power(preds - truths, 2)))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(),
                          global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class Trainer(object):

    def __init__(
            self,
            name,  # name of this experiment
            opt,  # extra conf
            model,  # network
            discriminator=None,  # discriminator
            d_optimizer=None,  # optimizer
            g_optimizer=None,  # optimizer
            latent_dim=128,  # z latent feature
            ema_decay=None,  # if use EMA, set the decay
            lr_scheduler=None,  # scheduler
            metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
            local_rank=0,  # which GPU am I
            world_size=1,  # total num of GPUs
            device=None,  # device to use, usually setting to None is OK. (auto choose device)
            mute=False,  # whether to mute all print
            fp16=False,  # amp optimize level
            eval_interval=1,  # eval once every $ epoch
            max_keep_ckpt=2,  # max num of saved ckpts in disk
            workspace='workspace',  # workspace to save logs & ckpts
            best_mode='min',  # the smaller/larger result, the better
            use_loss_as_metric=True,  # use loss as the first metric
            report_metric_at_train=False,  # also report metrics at training
            use_checkpoint="latest",  # which ckpt to use at init time
            use_tensorboardX=True,  # whether to use tensorboard for logging
            scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.latent_dim = latent_dim
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.critic_iter = 5

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])
        self.model = model

        if discriminator is not None:
            discriminator.to(self.device)
            self.discriminator = discriminator

        # if isinstance(criterion, nn.Module):
        #     criterion.to(self.device)
        # self.criterion = criterion

        if g_optimizer is not None:
            self.g_optimizer = g_optimizer(self.model)
            self.g_lr_scheduler = lr_scheduler(self.g_optimizer)

        if d_optimizer is not None and self.discriminator is not None:
            self.d_optimizer = d_optimizer(self.discriminator)
            self.d_lr_scheduler = lr_scheduler(self.d_optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(),
                                                decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints":
            [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}'
        )

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(
                        f"[INFO] {self.best_path} not found, loading latest ..."
                    )
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # clip loss prepare
        if opt.rand_pose >= 0:  # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf_gan.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text(
                [self.opt.clip_text])  # only support one text prompt now...

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def calculate_gradient_penalty(self, real_images, fake_images, real_poses,
                                   fake_poses):
        B = real_images.shape[0]
        eta = torch.FloatTensor(B).uniform_(0, 1)
        eta_images = eta.reshape(B, 1, 1).expand(B, real_images.size(1),
                                                 real_images.size(2))
        eta_images = eta_images.cuda()
        interpolated_images = eta_images * real_images + (
            (1 - eta_images) * fake_images)
        interpolated_images = interpolated_images.cuda()

        eta_poses = eta.reshape(B, 1, 1).expand(B, real_poses.size(1),
                                                real_poses.size(2))
        eta_poses = eta_poses.cuda()
        interpolated_poses = eta_poses * real_poses + (
            (1 - eta_poses) * fake_poses)
        interpolated_poses = interpolated_poses.cuda()

        # define it to calculate gradient
        interpolated_images.requires_grad_()
        interpolated_poses.requires_grad_()

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated_images,
                                               interpolated_poses)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=[interpolated_images, interpolated_poses],
            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
            create_graph=True,
            retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * 10.0
        return grad_penalty

    def train_step(self, data, z):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        # images = data['images']  # [B, N, 3/4]

        # B, N, C = images.shape

        # train in srgb color space
        bg_color = None
        # gt_rgb = images

        outputs = self.model.render(z,
                                    rays_o,
                                    rays_d,
                                    staged=False,
                                    bg_color=bg_color,
                                    perturb=True,
                                    **vars(self.opt))

        # pred_rgb = outputs['image']

        # loss = self.criterion(pred_rgb,
        #                       gt_rgb).mean(-1)  # [B, N, 3] --> [B, N]

        # loss = loss.mean()

        return outputs

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, z, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        H, W = data['H'], data['W']

        outputs = self.model.render(z,
                                    rays_o,
                                    rays_d,
                                    staged=True,
                                    bg_color=bg_color,
                                    perturb=perturb,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes',
                                     f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3],
                                               self.model.aabb_infer[3:],
                                               resolution=resolution,
                                               threshold=threshold,
                                               query_func=query_func)

        mesh = trimesh.Trimesh(
            vertices, triangles,
            process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        # if self.model.cuda_ray:
        #     self.model.mark_untrained_grid(train_loader._data.poses,
        #                                    train_loader._data.intrinsics)

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def test(self, loader, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format=
            '{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.model.eval()
        with torch.inference_mode():

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):
                for key in ['rays_o', 'rays_d', 'real_poses', 'fake_poses']:
                    data[key] = data[key].to(self.device)
                z = torch.rand((1, self.latent_dim))
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data, z)

                path = os.path.join(save_path, f'{i:04d}.png')
                path_depth = os.path.join(save_path, f'{i:04d}_depth.png')

                #self.log(f"[INFO] saving test image to {path}")

                cv2.imwrite(
                    path,
                    cv2.cvtColor((preds[0].detach().cpu().numpy() *
                                  255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(path_depth,
                            (preds_depth[0].detach().cpu().numpy() *
                             255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format=
                '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

        self.local_step = 0

        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        raster_settings = RasterizationSettings(
            image_size=256,
            bin_size=0,
            blur_radius=0.0,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(lights=lights, device=self.device))

        for it, data in enumerate(loader):
            for key in [
                    'rays_o', 'rays_d', 'rays_inds', 'mesh', 'real_poses',
                    'fake_poses'
            ]:
                data[key] = data[key].to(self.device)
            with torch.inference_mode():
                mesh = data['mesh'].extend(data['real_poses'].shape[0])
                poses = data['real_poses']
                cameras = PerspectiveCameras(
                    focal_length=(data['fx'] + data['fy']) / 2,
                    principal_point=((0.0, 0.0), ),
                    R=poses[:, :3, :3].transpose(1, 2),
                    T=-torch.bmm(poses[:, :3, :3].transpose(1, 2),
                                 poses[:, :3, 3:4])[:, :, 0],
                    device=self.device)
                images = renderer(mesh, cameras=cameras)[..., :3]
                B = images.shape[0]
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1,
                                      torch.stack(C * [data['rays_inds']],
                                                  -1))  # [B, N, 3/4]
                # img = images[0].reshape(32, 32, 3)
                # plt.figure(figsize=(10, 10))
                # plt.imshow(img.cpu().numpy())
                # plt.axis("off")
                # plt.show()
                data['images'] = images

            self.local_step += 1
            self.global_step += 1

            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.discriminator.parameters():
                p.requires_grad = True
            self.d_optimizer.zero_grad()
            z = torch.rand((1, self.latent_dim))
            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = self.train_step(data, z)
                d_loss_real = self.discriminator(data['images'],
                                                 data['real_poses']).mean()
                d_loss_fake = -self.discriminator(outputs['image'],
                                                  data['fake_poses']).mean()
                gradient_penalty = self.calculate_gradient_penalty(
                    data['images'], outputs['image'].data, data['real_poses'],
                    data['fake_poses'])
            d_loss = d_loss_real + d_loss_fake + gradient_penalty
            self.scaler.scale(d_loss_real).backward()
            self.scaler.scale(d_loss_fake).backward()
            self.scaler.scale(gradient_penalty).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()

            d_loss_val = d_loss.item()
            total_loss += d_loss_val

            for p in self.model.parameters():
                p.requires_grad = True
            for p in self.discriminator.parameters():
                p.requires_grad = False
            self.g_optimizer.zero_grad()
            z = torch.rand((1, self.latent_dim))
            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = self.train_step(data, z)
                g_loss = self.discriminator(outputs['image'],
                                            data['fake_poses'])
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
            g_loss_val = g_loss.item()

            total_loss += g_loss_val

            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                # for metric in self.metrics:
                #     metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/d_loss", d_loss_val,
                                           self.global_step)
                    self.writer.add_scalar("train/g_loss", g_loss_val,
                                           self.global_step)
                    self.writer.add_scalar(
                        "train/d_lr", self.d_optimizer.param_groups[0]['lr'],
                        self.global_step)
                    self.writer.add_scalar(
                        "train/g_lr", self.g_optimizer.param_groups[0]['lr'],
                        self.global_step)

                pbar.set_description(
                    f"d_loss={d_loss_val:.4f} g_loss={g_loss_val:.4f} ({total_loss/self.local_step:.4f})"
                )
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def save_checkpoint(self, full=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['d_optimizer'] = self.d_optimizer.state_dict()
            state['g_optimizer'] = self.g_optimizer.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        # if not best:
        state['model'] = self.model.state_dict()
        state['discriminator'] = self.discriminator.state_dict()

        file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

        self.stats["checkpoints"].append(file_path)

        if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
            old_ckpt = self.stats["checkpoints"].pop(0)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(
                glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log(
                    "[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
