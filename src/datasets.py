import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from glob import glob
import numpy as np
import h5py
from .utils import (
    get_mlp_params_as_matrix,
    flatten_mlp_params,
    unflatten_mlp_params,
    select_sampling_method_online,
)

from .models import ResnetFC

from tqdm import tqdm, trange


# Reshape point cloud such that it lies in bounding box of (-1, 1) * scale
def normalize_points(coords, scale=0.9):
    coords -= torch.mean(coords, dim=0, keepdim=True)
    coord_max, coord_min = coords.aminmax()
    coords = (coords - coord_min) / (coord_max - coord_min)  # (0, 1)
    coords = (coords - 0.5) * (2.0 * scale)
    return coords


class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        # point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        # point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface


"""

Various helper functions for permutation augmentation of neural networks.

Borrowed from https://github.com/wpeebles/G.pt

Although we're not using augmentation in final paper, it's here for reference
"""
import torch
from torch import nn


def permute_out(tensors, permutation):
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    for tensor in tensors:
        tensor.copy_(tensor[permutation])


def permute_in(tensors, permutation):
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    for tensor in tensors:
        tensor.copy_(tensor[:, permutation])


def permute_in_out(tensor, permutation_in, permutation_out):
    tensor.copy_(tensor[:, permutation_in][permutation_out])


def random_permute_flat(nets, architecture, seed, permutation_fn):
    """
    Applies an output-preserving parameter permutation to a list of nets, eah with shape (D,).
    The same permutation is applied to each network in the list.

    nets: list of torch.Tensor or torch.Tensor, each of shape (D,)
    """

    input_is_tensor = isinstance(nets, torch.Tensor)
    if input_is_tensor:
        nets = [nets]
    full_permute = []
    total = 0

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    def build_index(tensors, permutation, permutation2=None, permute=None):
        nonlocal total
        if isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        for tensor in tensors:
            num_params = tensor.numel()
            shape = tensor.shape
            indices = torch.arange(total, num_params + total).view(shape)
            if permute == "out":
                indices = indices[permutation]
            elif permute == "in":
                indices = indices[:, permutation]
            elif permute == "both":
                indices = indices[:, permutation][permutation2]
            elif permute == "none":
                pass
            total += num_params
            full_permute.append(indices.flatten())

    build_in_fn = lambda *args: build_index(*args, permute="in")
    build_out_fn = lambda *args: build_index(*args, permute="out")
    build_both_fn = lambda *args: build_index(*args, permute="both")
    register_fn = lambda x: build_index(x, permutation=None, permute="none")
    permutation_fn(
        architecture, generator, build_in_fn, build_out_fn, build_both_fn, register_fn
    )

    full_permute = torch.cat(full_permute)
    assert total == full_permute.size(0) == nets[0].size(0)
    permuted_nets = [
        net[full_permute] for net in nets
    ]  # Apply the same permutation to each net
    if input_is_tensor:  # Unpack the list to return in same format as input
        permuted_nets = permuted_nets[0]
    return permuted_nets


def random_permute_mlp(
    net,
    generator=None,
    permute_in_fn=permute_in,
    permute_out_fn=permute_out,
    permute_in_out_fn=permute_in_out,
    register_fn=lambda x: x,
):
    # NOTE: when using this function as part of random_permute_flat, THE ORDER IN WHICH
    # PERMUTE_OUT_FN, PERMUTE_IN_FN, etc. get called IS REALLY IMPORTANT. The order MUST be consistent
    # with whatever net.state_dict().keys() returns, otherwise the permutation will be INCORRECT.
    # If you're using this function directly on an MLP instance and then flattening its weights,
    # the order does NOT matter since everything is being done in-place.
    # assert isinstance(net, MLP3D) or isinstance(net, MLP) or isinstance(net, modules.SingleBVPNet)
    running_permute = None  # Will be set by initial nn.Linear

    linears = [module for module in net.modules() if isinstance(module, nn.Linear)]
    n_linear = len(linears)
    for ix, linear in enumerate(linears):
        if ix == 0:  # Input layer
            running_permute = torch.randperm(linear.weight.size(0), generator=generator)
            permute_out_fn((linear.weight, linear.bias), running_permute)
        elif ix == n_linear - 1:  # Output layer
            permute_in_fn(linear.weight, running_permute)
            register_fn(linear.bias)
        else:  # Hidden layers:
            new_permute = torch.randperm(linear.weight.size(0), generator=generator)
            permute_in_out_fn(linear.weight, running_permute, new_permute)
            permute_out_fn(linear.bias, new_permute)
            running_permute = new_permute


def sorted_permute_mlp(
    net,
    generator=None,
    permute_in_fn=permute_in,
    permute_out_fn=permute_out,
    permute_in_out_fn=permute_in_out,
    register_fn=lambda x: x,
):
    # NOTE: when using this function as part of random_permute_flat, THE ORDER IN WHICH
    # PERMUTE_OUT_FN, PERMUTE_IN_FN, etc. get called IS REALLY IMPORTANT. The order MUST be consistent
    # with whatever net.state_dict().keys() returns, otherwise the permutation will be INCORRECT.
    # If you're using this function directly on an MLP instance and then flattening its weights,
    # the order does NOT matter since everything is being done in-place.
    # assert isinstance(net, MLP3D) or isinstance(net, MLP) or isinstance(net, modules.SingleBVPNet)
    running_permute = None  # Will be set by initial nn.Linear

    linears = [module for module in net.modules() if isinstance(module, nn.Linear)]
    n_linear = len(linears)
    for ix, linear in enumerate(linears):
        if ix == 0:  # Input layer
            # running_permute = torch.randperm(linear.weight.size(0), generator=generator)
            running_permute = linear.weight.mean(dim=1).argsort()
            # print(running_permute[:10])
            permute_out_fn((linear.weight, linear.bias), running_permute)
        elif ix == n_linear - 1:  # Output layer
            permute_in_fn(linear.weight, running_permute)
            register_fn(linear.bias)
        else:  # Hidden layers:
            # new_permute = torch.randperm(linear.weight.size(0), generator=generator)
            new_permute = linear.weight.mean(dim=1).argsort()
            permute_in_out_fn(linear.weight, running_permute, new_permute)
            permute_out_fn(linear.bias, new_permute)
            running_permute = new_permute
        # print(ix, running_permute[:10])


class VascWeightDataset(Dataset):
    def __init__(self, opt, inrs_root, split, sample_sd, transform=None):
        self.inrs_root = inrs_root
        self.opt = opt
        self.sample_sd = ResnetFC(d_in=3, d_latent=0, d_out=1).state_dict()
        self.split = split
        files = np.loadtxt(
            str(self.inrs_root.parent / "vasc_splits" / f"{split}_split_1.txt"),
            dtype=str,
        )
        files = sorted(files.tolist())
        self.mlps_paths = [str(self.inrs_root / x) for x in files]
        # self.mlps_paths = sorted(self.inrs_root.glob("*.pth"))
        load_fn = lambda x: torch.load(x, map_location="cpu")
        self.mlps = [load_fn(x) for x in tqdm(self.mlps_paths)]
        self.vasc_names = ["_".join(x.split("_")[:2]) for x in files]
        self.intra_names = [x.split("_")[0] for x in files]

        self.intra_objs = [
            self.inrs_root / "processed_sdfs" / x for x in self.intra_names
        ]
        self.vasc_objs = [
            self.inrs_root / "processed_sdfs" / x for x in self.vasc_names
        ]
        dataroot1 = "/localscratch/asa409/intra_vessels/vessels"
        dataroot = "/localscratch/asa409/watertight_pifu_new/"
        self.all_files = [
            Path(x) for x in glob(str(Path(dataroot) / "CACHE" / "*.npz"))
        ]  # vasc
        # self.augment = transform
        if self.opt.augment in ["permute", "permute_same", "sort_permute"]:
            # self.example_mlp = get_mlp(mlp_kwargs)
            self.example_mlp = ResnetFC(d_in=3, d_latent=0, d_out=1).state_dict()
        self.augment = self.opt.augment

    def __len__(self):
        return len(self.mlps_paths)

    def get_weights(self, weights):
        # weights = torch.hstack(weights)
        prev_weights = weights.clone()
        if self.augment == "permute":
            weights = random_permute_mlp(
                [weights], self.example_mlp, None, random_permute_mlp
            )[0]
            # weights = flatten_mlp_params(self.example_mlp)
        return weights, prev_weights

    def __getitem__(self, idx):
        wts = self.mlps[idx]
        params = flatten_mlp_params(wts)
        matrix = get_mlp_params_as_matrix(params, self.sample_sd)

        name = self.vasc_names[idx]
        wts, prev_wts = self.get_weights(params)
        return wts.float(), prev_wts.float(), prev_wts.float(), matrix.float(), name


class IntraWeightDataset(Dataset):
    def __init__(self, opt, inrs_root, split, sample_sd, transform=None):
        self.inrs_root = inrs_root
        self.opt = opt
        self.sample_sd = ResnetFC(d_in=3, d_latent=0, d_out=1).state_dict()
        self.split = split
        files = np.loadtxt(
            str(self.inrs_root.parent / "intra_splits" / f"{split}_split_1.txt"),
            dtype=str,
        )
        files = sorted(files.tolist())
        self.mlps_paths = [str(self.inrs_root / x) for x in files]
        # self.mlps_paths = sorted(self.inrs_root.glob("*.pth"))
        load_fn = lambda x: torch.load(x, map_location="cpu")
        self.mlps = [load_fn(x) for x in tqdm(self.mlps_paths)]
        self.vasc_names = ["_".join(x.split("_")[:2]) for x in files]
        self.intra_names = [x.split("_")[0] for x in files]

        self.intra_objs = [
            self.inrs_root / "processed_sdfs" / x for x in self.intra_names
        ]
        self.vasc_objs = [
            self.inrs_root / "processed_sdfs" / x for x in self.vasc_names
        ]
        dataroot1 = "/localscratch/asa409/intra_vessels/vessels"
        dataroot = "/localscratch/asa409/watertight_pifu_new/"
        self.all_files = [
            Path(x) for x in glob(str(Path(dataroot1) / "*.obj"))
        ]  # intra
        self.augment = transform

    def __len__(self):
        return len(self.mlps_paths)

    def get_weights(self, weights):
        # weights = torch.hstack(weights)
        prev_weights = weights.clone()

        return weights, prev_weights

    def __getitem__(self, idx):
        wts = self.mlps[idx]
        params = flatten_mlp_params(wts)
        matrix = get_mlp_params_as_matrix(params, self.sample_sd)

        name = self.intra_names[idx]
        # file = self.all_files[idx]
        wts, prev_wts = self.get_weights(params)
        return wts.float(), prev_wts.float(), prev_wts.float(), matrix.float(), name
