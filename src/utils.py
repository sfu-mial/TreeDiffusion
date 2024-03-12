from torch import Tensor
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import functools

import mesh2sdf
import os
import time


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
