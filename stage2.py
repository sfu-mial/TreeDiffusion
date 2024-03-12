import os


from src.datasets import (
    IntraWeightDataset,
    VascWeightDataset,
    sorted_permute_mlp,
    random_permute_mlp,
    random_permute_flat,
)

# Using it to make pyrender work on clusters
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
from datetime import datetime
from os.path import join

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
import argparse
import warnings

warnings.filterwarnings("ignore")

import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, random_split
import util.lr_decay as lrd
import util.misc as misc
from glob import glob
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from glob import glob
import wandb
from tqdm import tqdm, trange
from datetime import datetime
import pytorch_lightning as pl
import sys
from src.trainer import DiffusionTrainer
from src.utils import (
    select_sampling_method_online,
    get_mlps_batched_params,
    flatten_mlp_params,
    unflatten_mlp_params,
    Config,
    calculate_fid_3d,
)

import mcubes
import trimesh

sys.path.append("siren")

from src.models import ResnetFC


def run(args):
    if args.test_only and args.ckpt_path is not None:
        run_name = Path(args.ckpt_path).parent.stem.split(f"{args.dset_name}_")[-1]
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_logger = pl.loggers.WandbLogger(
        project="inr2vec",
        entity="sinashish",
        name=f"DM_{args.output_dir.stem}",
        tags=[args.dset_name],
        config=args,
    )
    print("wandb", wandb.run.name, wandb.run.id)
    slurm_plugin = pl.plugins.environments.SLURMEnvironment()
    if args.dset_name == "intra":
        dataset_train = IntraWeightDataset(
            args, args.data_path, split="train", sample_sd=None
        )
        dataset_val = IntraWeightDataset(
            args, args.data_path, split="val", sample_sd=None
        )
        dataset_test = IntraWeightDataset(
            args, args.data_path, split="test", sample_sd=None
        )

    elif args.dset_name == "vasc":
        dataset_train = VascWeightDataset(
            args, args.data_path, split="train", sample_sd=None
        )
        dataset_val = VascWeightDataset(
            args, args.data_path, split="val", sample_sd=None
        )
        dataset_test = VascWeightDataset(
            args, args.data_path, split="test", sample_sd=None
        )

    # initialize INR
    mlp = ResnetFC(d_in=3, d_latent=0, d_out=1)
    state_dict = mlp.state_dict()
    layers = []
    layer_names = []
    for l in state_dict:
        shape = state_dict[l].shape
        layers.append(np.prod(shape))
        layer_names.append(l)
    # initialize Diffusion Transformer
    trans_kwargs = dict(
        n_embd=args.n_embed,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        split_policy=args.split_policy,
        use_global_residual=args.use_global_residual,
        condition=args.condition,
    )
    model = Transformer(layers, layer_names, **trans_kwargs).cuda()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        prefetch_factor=2,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    print(
        "Train dataset length: {} | Val dataset length: {} | Test dataset length: {}".format(
            len(dataset_train), len(dataset_val), len(dataset_test)
        )
    )
    input_data = next(iter(data_loader_train))[0]
    print(
        "Input data shape, min, max:",
        input_data.shape,
        input_data.min(),
        input_data.max(),
    )
    mlp_kwargs = dict(
        mlp_type=cfg.mlp_type,
        model_type=cfg.mlp_type,
        output_type="occ",
        mlp_dims=cfg.mlp_dims,
        out_size=1,
        out_act="sigmoid",
    )
    # Initialize Trainer
    diffuser = DiffusionTrainer(
        model,
        dataset_train,
        dataset_val,
        dataset_test,
        mlp_kwargs,
        input_data.shape,
        method,
        args,
    )
    # Specify where to save checkpoints
    checkpoint_path = join(
        args.output_dir,
        "DM",
        f"{args.dset_name}_{run_name}",
    )
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    best_acc_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/1-NN-CD-acc",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-nn-model",
    )

    best_mmd_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/lgan_mmd-CD",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-mmd-model",  # {epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    )

    last_model_saver = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="last",  # -{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
        save_on_train_epoch_end=True,
    )
    ddp_strategy = pl.strategies.ddp.DDPStrategy(
        find_unused_parameters=False,
        process_group_backend="nccl",  # opt.dist_backend,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=args.epochs,
        precision=32,
        strategy=ddp_strategy,
        gradient_clip_val=2.0,
        logger=wandb_logger,
        default_root_dir=checkpoint_path,
        callbacks=[
            best_acc_checkpoint,
            best_mmd_checkpoint,
            last_model_saver,
            lr_monitor,
        ],
        plugins=[slurm_plugin] if os.getenv("SLURM_JOB_ID") else None,
        check_val_every_n_epoch=100,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accum_iter,
    )
    if not args.test_only:
        # If model_resume_path is provided (i.e., not None), the training will continue from that checkpoint
        trainer.fit(diffuser, data_loader_train, data_loader_val)
    # best_model_save_path is the path to saved best model
    print("testing on test set")
    trainer.test(
        diffuser,
        data_loader_test,
        ckpt_path=args.ckpt_path if args.test_only else None,
    )
    print("testing on val set")
    trainer.test(
        diffuser,
        data_loader_val,
        ckpt_path=args.ckpt_path if args.test_only else None,
    )
    print("testing on train set")
    trainer.test(
        diffuser,
        data_loader_train,
        ckpt_path=args.ckpt_path if args.test_only else None,
    )
    wandb_logger.finalize("Success")


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
        "--pcd_path",
        default="/localscratch/asa409/intra_vessels/INR_PROCESSED_DATA/VASC_INR_occ/messh.ply",
        type=str,
        help="path of 3D shape as .obj/.ply",
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

    run(args)
