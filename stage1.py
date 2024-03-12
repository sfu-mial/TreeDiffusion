import warnings

warnings.filterwarnings("ignore")
from mmcv import Config
import numpy as np
import torch
import torch.nn as nn
from glob import glob
import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
import trimesh
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_euler_angles

from tqdm import tqdm, trange
import argparse
import pytorch_lightning as pl
from src.models import ResnetFC
from src.utils import select_sampling_method_online
from pathlib import Path

pl.seed_everything(1997)

torch.cuda.manual_seed_all(1997)
np.random.seed(1997)
torch.cuda.manual_seed(1997)


def overfit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = ResnetFC(d_in=3, d_out=1, n_layers=args.n_blocks, d_hidden=args.d_hidden)
    mlp = mlp.to(device)

    optimizer = torch.optim.Adam(
        mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    filename = Path(args.data_path)
    out_root = Path(args.output_dir)
    out_root.mkdir(exist_ok=True, parents=True)
    print(f"Overfitting on {filename}")
    data = select_sampling_method_online(
        filename, num_sample_inout=args.nsample, is_train=True
    )
    save_name = out_root / f"{data['name']}_overfitted_model_fine.pth"
    for i in trange(args.epochs):
        pts = data["samples"][None].cuda().transpose(2, 1)
        gt = data["labels"][None].cuda().transpose(1, 2)
        optimizer.zero_grad()
        pred = model(pts.cuda()).sigmoid().squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, gt.squeeze(-1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Mesh: {data['name']} | Iter: {i} | Loss: {loss.item()}")
        if i % 500 == 0:
            try:
                with torch.no_grad():
                    model.eval()
                    mesh, sdf = extract_mesh_from_mlp(model, N=128)
                    model.train()
                    mesh.export(
                        out_root / f"{data['name']}_pred_128_iter_{i+1}_fine.ply"
                    )
            except Exception as e:
                print(f"Error in exporting mesh: {e}")
            data = select_sampling_method_online(filename, 10000)
            torch.save(
                model.state_dict(),
                str(save_name),
            )
            print(f"{file} overfitted and saved to {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage 1")
    parser.add_argument(
        "--batch_size",
        "-bs",
        default=8,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--dset_name", default="vasc", type=str, choices=["vasc", "intra"]
    )
    parser.add_argument("--epochs", "-e", default=6000, type=int)
    parser.add_argument(
        "--accum_iter",
        "-ai",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--ckpt_path", default=None, help="checkpoint")
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=5e-5,
        help="weight decay (default: 0.05)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--inrs_root",
        "-i",
        type=str,
        default="/localscratch/asa409/intra_vessels/INR_PROCESSED_DATA/VASC_INR_occ/",
        help="path to the root directory of point cloud data",
    )
    parser.add_argument(
        "--data_path",
        default="/localscratch/asa409/intra_vessels/INR_PROCESSED_DATA/VASC_INR_occ/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--output_dir",
        default="/localscratch/asa409/intra_vessels/output/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output/", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=1997, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--method", default="hyper_3d", type=str)
    parser.add_argument("--beta_schedule", default="linear", type=str)
    parser.add_argument("--test_sample_mult", default=1.1, type=float)
    parser.add_argument(
        "--augment",
        default="",
        choices="permute/permute_same/sort_permute",
        type=str,
        help="augmentation method",
    )
    parser.add_argument("--augment_amount", default=0.0, type=float)
    parser.add_argument("--timesteps", default=500, type=int)
    parser.add_argument("--scheduler_step", "-ss", default=100, type=int)
    parser.add_argument("--sampling", default="ddim", type=str)
    parser.add_argument("--num_points", default=5000, type=int)
    parser.add_argument("--mlp_dims", default=128, type=int)
    parser.add_argument("--mlp_type", default="mlp", type=str)
    parser.add_argument(
        "--d_hidden", default=128, type=int, help="number of hidden dims of MLP"
    )
    parser.add_argument(
        "--n_blocks", default=5, type=int, help="number of resnet blocks"
    )
    parser.add_argument("--n_embed", default=128, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--n_heads", default=4, type=int)
    parser.add_argument("--condition", default="no", type=str)
    parser.add_argument("--split_policy", default="layer_by_layer", type=str)
    parser.add_argument("--use_global_residual", action="store_true")

    parser.add_argument(
        "--model_mean_type",
        default="START_X",
        type=str,
        choices=["START_X", "PREVIOUS_X", "EPSILON"],
    )
    parser.add_argument(
        "--model_var_type",
        default="FIXED_LARGE",
        type=str,
        choices=["LEARNED", "FIXED_LARGE", "FIXED_SMALL", "LEARNED_RANGE"],
    )
    parser.add_argument(
        "--loss_type",
        default="MSE",
        type=str,
        choices=["MSE", "RESCALED_MSE", "KL", "RESCALED_KL"],
    )
    return parser
    args = parser.parse_args()

    overfit(args)
