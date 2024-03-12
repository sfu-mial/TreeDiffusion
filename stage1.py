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
from src.utils import select_sampling_method_online, extract_mesh_from_mlp
from pathlib import Path


pl.seed_everything(1997)

torch.cuda.manual_seed_all(1997)
np.random.seed(1997)
torch.cuda.manual_seed(1997)


def overfit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResnetFC(d_in=3, d_out=1, n_blocks=args.n_blocks, d_hidden=args.d_hidden)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    filename = Path(args.pcd_path)
    out_root = Path(args.output_dir) / "preds"
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
            # sample new points every 500 iterations for robust fitting
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
            print(f"{filename} overfitted and saved to {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage 1")

    parser.add_argument("--epochs", "-e", default=6000, type=int)

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
        "--output_dir",
        default="./output/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=1997, type=int)
    parser.add_argument(
        "--d_hidden", default=128, type=int, help="number of hidden dims of MLP"
    )
    parser.add_argument(
        "--n_blocks", default=5, type=int, help="number of resnet blocks"
    )
    parser.add_argument(
        "--pcd_path",
        default="/path/to/mesh.ply",
        type=str,
        help="path of 3D shape as .obj/.ply",
    )
    args = parser.parse_args()

    overfit(args)
