from torch import Tensor
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import List, Dict, Any, Tuple, Union

import mesh2sdf
import os
import time

from math import ceil

import numpy as np
import pyrender
import torch
import trimesh

from .models import ResnetFC
from .Pointnet_Pointnet2_pytorch.log.classification.pointnet2_ssg_wo_normals import (
    pointnet2_cls_ssg,
)
from math import ceil

import numpy as np
import pyrender
import torch
import trimesh

from .torchmetrics_fid import FrechetInceptionDistance

model_url = "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"
model_save_path = "/home/asa409/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
saved_model_path = "/project/mial-lab/asa409/inr2vec/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"

import os

if not os.path.exists(model_save_path) and os.path.isfile(model_save_path):
    os.system(f"wget {model_url} -O {model_save_path}")

# Using edited 2D-FID code of torch_metrics
fid = FrechetInceptionDistance(
    reset_real_features=True, feature_extractor_weights_path=saved_model_path
)


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_mlp_params_as_matrix(flattened_params: Tensor, sd: Dict[str, Any]) -> Tensor:
    params_shapes = [p.shape for p in sd.values()]
    feat_dim = params_shapes[0][0]
    start = params_shapes[0].numel() + params_shapes[1].numel()
    if "lin_out.weight" in sd.keys():
        end = params_shapes[2].numel() + params_shapes[3].numel()
    else:
        end = params_shapes[-1].numel() + params_shapes[-2].numel()
    params = flattened_params[start:-end]
    return params.reshape((-1, feat_dim))


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Config:
    config = None

    @staticmethod
    def get(param):
        return Config.config[param] if param in Config.config else None


def calculate_fid_3d(
    sample_pcs,
    ref_pcs,
    wandb_logger,
    path="./Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth",
):
    batch_size = 10
    if not os.path.exists(path) and not os.path.exists(model_save_path):
        os.makedirs(
            "./Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/",
            exist_ok=True,
        )
        os.system(
            f"wget https://github.com/Rgtemze/HyperDiffusion/blob/main/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth -O {path}"
        )
    point_net = pointnet2_cls_ssg.get_model(40, normal_channel=False)
    try:
        checkpoint = torch.load(path, map_location="cuda")
    except:
        checkpoint = torch.load(saved_model_path, map_location="cuda")
        print("model loaded from pointnet")
    point_net.load_state_dict(checkpoint["model_state_dict"])
    point_net.eval().to(sample_pcs.device)
    count = len(sample_pcs)
    for i in range(ceil(count / batch_size)):
        if i * batch_size >= count:
            break
        real_features = point_net(
            ref_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fake_features = point_net(
            sample_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fid.update(real_features, real=True, features=real_features)
        fid.update(fake_features, real=False, features=fake_features)

    x = fid.compute()
    fid.reset()
    print("x fid_value", x)
    return x


def state_dict_to_weights(state_dict):
    weights = []
    for weight in state_dict:
        weights.append(state_dict[weight].flatten())
    weights = torch.hstack(weights)
    return weights


def generate_mlp_from_weights(weights, mlp_kwargs):
    mlp = ResnetFC(d_in=3, d_latent=0, d_out=1)
    state_dict = mlp.state_dict()
    weight_names = list(state_dict.keys())
    for layer in weight_names:
        val = state_dict[layer]
        num_params = np.product(list(val.shape))
        w = weights[:num_params]
        w = w.view(*val.shape)
        state_dict[layer] = w
        weights = weights[num_params:]
    assert len(weights) == 0, f"len(weights) = {len(weights)}"
    mlp.load_state_dict(state_dict)
    return mlp


def render_meshes(meshes):
    out_imgs = []
    for mesh in meshes:
        img, _ = render_mesh(mesh)
        out_imgs.append(img)
    return out_imgs


def render_mesh(obj):
    if isinstance(obj, trimesh.Trimesh):
        # Handle mesh rendering
        mesh = pyrender.Mesh.from_trimesh(
            obj,
            material=pyrender.MetallicRoughnessMaterial(
                alphaMode="BLEND",
                baseColorFactor=(0 / 255.0, 235 / 255.0, 193 / 255.0),
                # baseColorFactor=[0.8, 0.3, 0.3, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.5,
            ),
        )
    else:
        # Handle point cloud rendering, (converting it into a mesh instance)
        pts = obj
        sm = trimesh.creation.uv_sphere(radius=0.01)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts
        mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    eye = np.array([2, 1.4, -2])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    camera_pose = look_at(eye, target, up)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(256, 256)
    color, depth = r.render(scene)
    r.delete()
    return color, depth


# Calculate look-at matrix for rendering
def look_at(eye, target, up):
    forward = eye - target
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    camera_pose = np.eye(4)
    camera_pose[:-1, 0] = right
    camera_pose[:-1, 1] = up
    camera_pose[:-1, 2] = forward
    camera_pose[:-1, 3] = eye
    return camera_pose


def get_mlps_batched_params(mlps: List) -> List[Tensor]:
    params = []
    for i in range(len(mlps)):
        params.append(list(mlps[i].parameters()))

    batched_params = []
    for i in range(len(params[0])):
        p = torch.stack([p[i] for p in params], dim=0)
        p = torch.clone(p.detach())
        p.requires_grad = True
        batched_params.append(p)

    return batched_params


def flatten_mlp_params(sd: Dict[str, Any]) -> Tensor:
    all_params = []
    for k in sd:
        all_params.append(sd[k].view(-1))
    all_params = torch.cat(all_params, dim=-1)
    return all_params


def unflatten_mlp_params(
    params: Tensor,
    sample_sd: Dict[str, Any],
) -> Dict[str, Any]:
    sd = collections.OrderedDict()

    start = 0
    for k in sample_sd:
        end = start + sample_sd[k].numel()
        layer_params = params[start:end].view(sample_sd[k].shape)
        sd[k] = layer_params
        start = end

    return sd


def select_sampling_method_online(cache_path, num_sample_inout=10000, is_train=True):
    if not is_train:
        random.seed(1991)
        np.random.seed(1991)
        torch.manual_seed(1991)

    def _subsample(points, labels, num):
        idx = np.random.randint(0, len(points), num)
        return points[idx], labels[idx]

    def _shuffle(sample_points, inside):
        index = list(range(len(sample_points)))
        np.random.shuffle(index)

        return sample_points[index], inside[index]

    subject = cache_path.stem.split("_fixed")[0]
    mesh = trimesh.load(cache_path, process=False, maintain_order=True)

    length = max(mesh.bounding_box.extents)
    center = mesh.bounding_box.centroid
    scale = 2 / length
    mesh.vertices -= center
    mesh.vertices *= scale

    nvol = 2 * num_sample_inout  # int(num_sample_inout * 0.1)
    nrand = num_sample_inout // 4  # int(num_sample_inout * 0.1)
    nsurf = 4 * num_sample_inout  # - nvol - nrand

    surf = mesh.sample(nsurf)
    surface_points = surf + np.random.normal(scale=0.01, size=surf.shape)
    surface_points1 = surf + np.random.normal(scale=0.03, size=surf.shape)
    surface_points2 = surf + np.random.normal(scale=0.02, size=surf.shape)
    surface_inside = mesh.contains(surface_points)
    surface_inside1 = mesh.contains(surface_points1)
    surface_inside2 = mesh.contains(surface_points2)

    volume_points = trimesh.sample.volume_mesh(mesh, nsurf)
    volume_inside = mesh.contains(volume_points)
    random_points = np.random.rand(*surf.shape) * 2 - 1
    random_inside = mesh.contains(random_points)

    sample_points = np.concatenate(
        (surface_points, surface_points1, surface_points2), axis=0
    )
    sample_inside = np.concatenate(
        (surface_inside, surface_inside1, surface_inside2), axis=0
    )
    volume_points, volume_inside = _subsample(volume_points, volume_inside, nvol)
    vp, vi = _subsample(volume_points, volume_inside, num_sample_inout)
    sp, si = _subsample(sample_points, sample_inside, num_sample_inout)
    vp = torch.from_numpy(vp.T).float()
    vl = torch.from_numpy(vi.reshape(1, -1)).float()
    sp = torch.from_numpy(sp.T).float()
    sl = torch.from_numpy(si.reshape(1, -1)).float()
    ###########################################################################
    ######### TODO: select a sampling ratio between surface and random points #######
    ######### Choose from 16:1, 4:1, 2:1, 1:1 #######################################
    k = 4  # sampling ratio k = 16: 1 (4/ 1/4)
    # #########################################################
    sample_points, sample_label = _subsample(
        sample_points, sample_inside, k * num_sample_inout
    )
    random_points, random_label = _subsample(
        random_points, random_inside, num_sample_inout // k
    )
    # shuffle the data

    # inspired from https://github.com/YuliangXiu/ICON/blob/a45b0258c521945dcb8c739aecfb40e1e611265f/lib/dataset/PIFuDataset.py#L561

    sample_points = np.concatenate([sample_points, random_points], 0)
    inside = np.concatenate([sample_label, random_label], 0)

    sample_points, inside = _shuffle(sample_points, inside)

    # breakpoint()
    # print (sample_points.shape, inside.shape, inside.dtype)
    # TODO: change to threshold value
    inside_points = sample_points[inside >= 0.5]
    outside_points = sample_points[inside < 0.5]

    # the number of inside points
    nin = inside_points.shape[0]
    inside_points = (
        inside_points[: num_sample_inout // 2]
        if nin > num_sample_inout // 2
        else inside_points
    )
    outside_points = (
        outside_points[: num_sample_inout // 2]
        if nin > num_sample_inout // 2
        else outside_points[: (num_sample_inout - nin)]
    )

    samples = np.concatenate([inside_points, outside_points], 0).T
    labels = np.concatenate(
        [
            np.ones((1, inside_points.shape[0])),
            np.zeros((1, outside_points.shape[0])),
        ],
        1,
    )
    samples = torch.from_numpy(samples).float()
    labels = torch.from_numpy(labels).float()

    return {
        "samples": samples,
        "labels": labels,
        "volume_points": vp,
        "volume_inside": vl,
        "surface_points": sp,
        "surface_inside": sl,
        # "b_min": b_min,
        # "b_max": b_max,
        "scale": scale,
        "center": center,
        "name": subject,
    }


def plot_mesh_plt(mesh, save_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(90, 270)
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        cmap="Spectral",
    )
    plt.savefig(save_name.replace(".ply", ".png"))
    plt.show()


def make_watertight(filename):
    mesh_scale = 0.8
    size = 128
    level = 2 / size

    mesh = trimesh.load(filename, force="mesh")

    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    # fix mesh
    t0 = time.time()
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True
    )
    t1 = time.time()

    # output
    mesh.vertices = mesh.vertices / scale + center
    mesh.export(filename[:-4] + "_fixed.obj")
    np.save(filename[:-4] + ".npy", sdf)
    print("It takes %.4f seconds to process %s" % (t1 - t0, filename))


@torch.no_grad()
def extract_mesh_from_mlp(model, N=128, thres=0.5):
    from mcubes import marching_cubes

    model.eval()
    x = torch.linspace(-1, 1, N)
    grid = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), -1).cuda().reshape(-1, 3)
    res_all = []
    padding = 3
    with torch.no_grad():
        for pts in torch.split(grid, 32**3, 0):
            pred = model(pts).sigmoid().squeeze()
            res_all.append(pred.detach())
        res_all = torch.cat(res_all, 0).reshape(N, N, N).cpu().numpy()
        padded_data = np.pad(res_all, padding, mode="constant", constant_values=0)
    v, f = marching_cubes(padded_data, thres)
    v -= padding
    mesh = trimesh.Trimesh(v, f)
    if v.shape[0] > 0:
        mesh.vertices -= mesh.bounding_box.centroid
        mesh.vertices *= mesh.bounding_box.extents.max()
        return mesh, res_all
    else:
        print("Empty Mesh")
        return None, padded_data
