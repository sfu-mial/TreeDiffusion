import copy
import os
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
from models import ResnetFC
from utils import (
    flatten_mlp_params,
    get_mlp_params_as_matrix,
    select_sampling_method_online,
)

import mcubes
import trimesh

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh

from scipy.spatial.transform import Rotation
from tqdm import tqdm

import wandb
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from evaluation_metrics_3d import compute_all_metrics, compute_all_metrics_4d
from utils import (
    calculate_fid_3d,
    Config,
    generate_mlp_from_weights,
    render_mesh,
    render_meshes,
)
from mcubes import marching_cubes


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionTrainer(pl.LightningModule):
    def __init__(
        self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, cfg
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.cfg.dedup = False
        self.cfg.calculate_metric_on_test = True
        self.cfg.val_fid_calculation_period = 15
        self.cfg.val_num_samples = 2
        self.cfg.val_num_points = 2048
        self.method = method
        self.mlp_kwargs = mlp_kwargs
        self.cfg.normalization_factor = 1
        self.val_dt = val_dt
        self.train_dt = train_dt
        self.test_dt = test_dt
        self.ae_model = None
        self.sample_count = min(
            8, self.cfg.batch_size
        )  # it shouldn't be more than 36 limited by batch_size
        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        print("encoded_outs.shape", encoded_outs.shape)
        timesteps = self.cfg.timesteps
        betas = get_named_beta_schedule(self.cfg.beta_schedule, timesteps)
        self.image_size = encoded_outs[:1].shape
        self.save_hyperparameters()

        Config.config = cfg
        # Initialize diffusion utiities
        # breakpoint()
        self.diff = GaussianDiffusion(
            betas=betas,
            # cfg=cfg,
            model_mean_type=ModelMeanType[cfg.model_mean_type],
            model_var_type=ModelVarType[cfg.model_var_type],
            loss_type=LossType[cfg.loss_type],
            diff_pl_module=self,
        )

    def forward(self, images):
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(images.shape[0],))
            .long()
            .to(self.device)
        )
        images = images * self.cfg.normalization_factor
        x_t, e = self.diff.q_sample(images, t)
        x_t = x_t.float()
        e = e.float()
        return self.model(x_t, t), e

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        # if self.cfg.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.scheduler_step, gamma=0.9
        )
        return [optimizer], [scheduler]
        # return optimizer

    def training_step(self, train_batch, batch_idx):
        # Extract input_data (either voxel or weight) which is the first element of the tuple
        input_data = train_batch[0]
        name = train_batch[-1]
        shape_id = int(name.split("_data")[-1])

        self.mlp_kwargs = dict(
            mlp_type=self.cfg.mlp_type,
            model_type=self.cfg.mlp_type,
            output_type="occ",
            mlp_dims=self.cfg.mlp_dims,
            out_size=1,
            out_act="sigmoid",
        )
        # Sample a diffusion timestep
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(input_data.shape[0],))
            .long()
            .to(self.device)
        )

        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            input_data * self.cfg.normalization_factor,
            t,
            self.mlp_kwargs,
            self.logger,
            model_kwargs=None,
        )
        loss_mse = loss_terms["loss"].mean()
        self.log("train_loss", loss_mse)

        loss = loss_mse
        return loss

    def validation_step(self, val_batch, batch_idx):
        metric_fn = self.calc_metrics

        metrics = metric_fn("train")
        for metric_name in metrics:
            self.log("train/" + metric_name, metrics[metric_name])
        metrics = metric_fn("val")
        for metric_name in metrics:
            self.log("val/" + metric_name, metrics[metric_name])

    def training_epoch_end(self, outputs) -> None:
        epoch_loss = sum(output["loss"] for output in outputs) / len(outputs)
        self.log("epoch_loss", epoch_loss)
        # Handle 3D generation
        if (self.current_epoch + 1) % 100 == 0 and self.global_rank == 0:
            x_0s = (
                self.diff.ddim_sample_loop(self.model, (4, *self.image_size[1:]))
                .cpu()
                .float()
            )
            x_0s = x_0s / self.cfg.normalization_factor
            print(
                "x_0s[0].stats",
                x_0s.min().item(),
                x_0s.max().item(),
                x_0s.mean().item(),
                x_0s.std().item(),
            )
            if False:  # self.mlp_kwargs.move:
                pass
            else:
                meshes, sdfs = self.generate_meshes(x_0s, None, res=64)
                for mesh in meshes:
                    if mesh.vertices.shape[0] > 0:
                        c = mesh.bounding_box.centroid
                        l = mesh.bounding_box.extents.max()
                        s = 2 / l
                        mesh.vertices -= c
                        mesh.vertices *= s
                print(
                    "sdfs.stats",
                    sdfs.min().item(),
                    sdfs.max().item(),
                    sdfs.mean().item(),
                    sdfs.std().item(),
                )
                out_imgs = render_meshes(meshes)
                self.logger.log_image(
                    "generated_renders", out_imgs, step=self.current_epoch
                )

    @torch.no_grad()
    def run_marching_cubes(self, model, N, thres=0.0):
        model.eval()
        x = torch.linspace(-1, 1, N)
        grid = (
            torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1)
            .to(self.device)
            .reshape(-1, 3)
        )
        res_all = []
        for pts in torch.split(grid, 10000, 0):
            res = model(pts).sigmoid().squeeze()
            res_all.append(res.detach().cpu())

        res_all = torch.cat(res_all, 0).cpu().numpy().reshape(N, N, N)
        # apply padding to avoid numerical issues
        pad_width = 5
        res_all = np.pad(res_all, pad_width, mode="constant", constant_values=1.0)
        v, f = marching_cubes(res_all, thres)
        v -= pad_width
        v /= N - 1
        v = v * 2 - 1
        return v, f, torch.from_numpy(res_all).to(self.device)

    def generate_meshes(self, x_0s, folder_name="meshes", info="0", res=64, level=0.5):
        x_0s = x_0s.view(len(x_0s), -1)
        curr_weights = Config.get("curr_weights")
        x_0s = x_0s[:, :curr_weights]
        meshes = []
        sdfs = []
        for i, weights in enumerate(x_0s):
            sdf_decoder = generate_mlp_from_weights(weights, self.mlp_kwargs)
            sdf_decoder.to(self.device).eval()
            with torch.no_grad():
                effective_file_name = (
                    f"{folder_name}/mesh_epoch_{self.current_epoch}_{i}_{info}"
                    if folder_name is not None
                    else None
                )
                v, f, sdf = self.run_marching_cubes(
                    sdf_decoder,
                    res,
                    thres=(
                        level
                        if self.mlp_kwargs["output_type"] in ["occ", "logits"]
                        else 0
                    ),
                )

                if (
                    "occ" in self.mlp_kwargs["output_type"]
                    or "logits" in self.mlp_kwargs["output_type"]
                ):
                    tmp = copy.deepcopy(f[:, 1])
                    f[:, 1] = f[:, 2]
                    f[:, 2] = tmp
                sdfs.append(sdf)
                mesh = trimesh.Trimesh(v, f)
                mesh = mesh.simplify_quadric_decimation(30000)
                # mesh.vertices -= mesh.bounding_box.centroid
                # mesh.vertices *= (2.0/mesh.bounding_box.extents.max())
                meshes.append(mesh)
        sdfs = torch.stack(sdfs)
        return meshes, sdfs

    def print_summary(self, flat, func):
        var = func(flat, dim=0)
        print(
            var.shape,
            var.mean().item(),
            var.std().item(),
            var.min().item(),
            var.max().item(),
        )
        print(var.shape, func(flat))

    def calc_metrics(self, split_type):
        self.cfg.inrs_root = Path(self.cfg.inrs_root)
        if "vasc" in str(self.cfg.inrs_root.stem).lower():
            files = np.loadtxt(
                str(
                    self.cfg.inrs_root.parent
                    / "vasc_splits"
                    / f"{split_type}_split_1.txt"
                ),
                dtype=str,
            ).tolist()
            files = sorted(files)
            import socket

            if "mial" in socket.gethostname():
                dataset_path = (
                    "/localhome/asa409/DATA/WT_vascusynth/VASCSYNTH_processed/CACHE/"
                )
            else:
                dataset_path = "/localscratch/asa409/watertight_pifu_new/CACHE/"
            test_object_names = ["_".join(x.split("_")[:2]) for x in files]
        else:
            files = np.loadtxt(
                str(
                    self.cfg.inrs_root.parent
                    / "intra_splits"
                    / f"{split_type}_split_1.txt"
                ),
                dtype=str,
            ).tolist()
            files = sorted(files)
            import socket

            if "mial" in socket.gethostname():
                dataset_path = "/localhome/asa409/DATA/brain_vessels/vessels/"
            else:
                dataset_path = "/localscratch/asa409/intra_vessels/vessels/"
            test_object_names = [x.split("_")[0] for x in files]

        orig_meshes_dir = (
            self.sample_dir
        )  # f"{self.cfg.output_dir}/orig_meshes/run_{wandb.run.name}"
        self.orig_meshes_dir = orig_meshes_dir
        os.makedirs(orig_meshes_dir, exist_ok=True)

        # During validation, only use some of the val and train shapes for speed
        if split_type == "val" and self.cfg.val_num_samples is not None:
            test_object_names = test_object_names[: self.cfg.val_num_samples]
        elif split_type == "train" and self.cfg.val_num_samples is not None:
            test_object_names = test_object_names[: self.cfg.val_num_samples]
        n_points = self.cfg.val_num_points

        # First process ground truth shapes
        pcs = []
        for obj_name in test_object_names:
            if self.cfg.dset_name == "vasc":
                pc = np.load(
                    os.path.join(dataset_path, obj_name + "_cache_all.npz"),
                    allow_pickle=True,
                )
                pc = pc["volume_points"]
            elif self.cfg.dset_name == "intra":
                pc = trimesh.load(
                    os.path.join(dataset_path, obj_name + "_fixed.obj")
                ).sample(10000)
                pc = np.array(pc).astype(float)
            pc = pc[:, :3]
            idx = np.arange(pc.shape[0])
            idx = np.random.choice(idx, self.cfg.val_num_points)
            pc = pc[idx]

            pc = torch.tensor(pc).float()
            pcs.append(pc)
        r = Rotation.from_euler("x", 90, degrees=True)
        ref_pcs = torch.stack(pcs)

        # We are generating slightly more than ref_pcs
        number_of_samples_to_generate = int(len(ref_pcs) * self.cfg.test_sample_mult)

        # Then process generated shapes
        sample_x_0s = []
        all_names = [[x] * int(self.cfg.test_sample_mult) for x in test_object_names]
        all_names = [item for sublist in all_names for item in sublist]
        test_batch_size = 100 if self.cfg.method == "hyper_3d" else self.cfg.batch_size

        for _ in tqdm(range(number_of_samples_to_generate // test_batch_size)):
            sample_x_0s.append(
                self.diff.ddim_sample_loop(
                    self.model, (test_batch_size, *self.image_size[1:])
                )
            )

        if number_of_samples_to_generate % test_batch_size != 0:
            sample_x_0s.append(
                self.diff.ddim_sample_loop(
                    self.model,
                    (
                        number_of_samples_to_generate % test_batch_size,
                        *self.image_size[1:],
                    ),
                )
            )

        sample_x_0s = torch.vstack(sample_x_0s)
        torch.save(sample_x_0s, f"{orig_meshes_dir}/prev_sample_x_0s.pth")
        print(sample_x_0s.shape)
        if self.cfg.dedup:
            sample_dist = torch.cdist(
                sample_x_0s,
                sample_x_0s,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            sample_dist_min = sample_dist.kthvalue(k=2, dim=1)[0]
            sample_dist_min_sorted = torch.argsort(sample_dist_min, descending=True)[
                : int(len(ref_pcs) * 1.01)
            ]
            sample_x_0s = sample_x_0s[sample_dist_min_sorted]
            print(
                "sample_dist.shape, sample_x_0s.shape",
                sample_dist.shape,
                sample_x_0s.shape,
            )
        torch.save(
            dict(sample=sample_x_0s, name=all_names),
            f"{orig_meshes_dir}/sample_x_0s.pth",
        )
        print("Sampled")

        print("Running marching cubes")
        sample_batch = []
        for x_0s in tqdm(sample_x_0s):
            mesh, _ = self.generate_meshes(
                x_0s.unsqueeze(0) / self.cfg.normalization_factor,
                self.sample_dir,
                res=128 if split_type == "test" else 128,
                level=0.5 if split_type == "test" else 0.5,
            )
            mesh = mesh[0]
            if len(mesh.vertices) > 0:
                pc = torch.tensor(mesh.sample(n_points))
                # if not self.cfg.mlp_config.params.move and "hyper" in self.cfg.method:
                pc = pc.float()
            else:
                print("Empty mesh")
                if split_type in ["val", "train"]:
                    pc = torch.zeros_like(ref_pcs[0])
                else:
                    print(f"Got empty mesh for {split_type}")
                    pc = torch.zeros_like(ref_pcs[0])
            sample_batch.append(pc)
        print("Marching cubes completed")
        print("number of samples generated:", len(sample_batch))
        sample_batch = sample_batch[: len(ref_pcs)]
        print("number of samples generated (after clipping):", len(sample_batch))
        sample_pcs = torch.stack(sample_batch)
        assert len(sample_pcs) == len(ref_pcs)
        torch.save(sample_pcs, f"{orig_meshes_dir}/samples.pth")

        self.logger.experiment.log(
            {"3d_gen": wandb.Object3D(r.apply(np.array(sample_pcs[0])))}
        )
        print("Starting metric computation for", split_type)
        fid = calculate_fid_3d(
            sample_pcs.to(self.device), ref_pcs.to(self.device), self.logger
        )
        metrics = compute_all_metrics(
            sample_pcs.to(self.device),
            ref_pcs.to(self.device),
            16 if split_type == "test" else 16,
            (self.logger, self.sample_dir),
        )
        metrics["fid"] = fid.item()

        print("Completed metric computation for", split_type)

        return metrics

    def sample_from_ddim(self, num_samples, res=128):
        metric_fn = self.calc_metrics("test")

        pass

    def test_step(self, *args, **kwargs):
        self.mlp_kwargs = dict(
            mlp_type=self.cfg.mlp_type,
            model_type=self.cfg.mlp_type,
            output_type="occ",
            mlp_dims=self.cfg.mlp_dims,
            out_size=1,
            out_act="sigmoid",
        )
        batch_idx = f"{args[-2][-1][0]}_{args[-1]}"
        os.makedirs(f"{self.cfg.output_dir}/gen_meshes/{batch_idx}", exist_ok=True)
        self.sample_dir = f"{self.cfg.output_dir}/gen_meshes/{batch_idx}"
        if self.cfg.calculate_metric_on_test:
            metric_fn = self.calc_metrics
            metrics = metric_fn("test")
            print("test", metrics)
            for metric_name in metrics:
                self.log("test_sample/" + metric_name, metrics[metric_name])
        if self.cfg.calculate_metric_on_test:
            metric_fn = self.calc_metrics
            metrics = metric_fn("train")
            print("train", metrics)
            for metric_name in metrics:
                self.log("train_sample/" + metric_name, metrics[metric_name])
        if self.cfg.calculate_metric_on_test:
            metric_fn = self.calc_metrics
            metrics = metric_fn("val")
            print("val", metrics)
            for metric_name in metrics:
                self.log("val_sample/" + metric_name, metrics[metric_name])

        x_0s = []
        all_names = []
        for i, img in enumerate(self.train_dt):
            x_0s.append(img[0])
            name = img[-1]
            all_names.append(name)
        x_0s = torch.stack(x_0s).to(self.device)
        flat = x_0s.view(len(x_0s), -1)
        # return
        print(x_0s.shape, flat.shape)
        print("Variance With zero-padding")
        self.print_summary(flat, torch.var)
        print("Variance Without zero-padding")
        self.print_summary(flat[:, : Config.get("curr_weights")], torch.var)

        print("Mean With zero-padding")
        self.print_summary(flat, torch.mean)
        print("Mean Without zero-padding")
        self.print_summary(flat[:, : Config.get("curr_weights")], torch.mean)

        stdev = x_0s.flatten().std(unbiased=True).item()
        oai_coeff = (
            0.538 / stdev
        )  # 0.538 is the variance of ImageNet pixels scaled to [-1, 1]
        print(f"Standard Deviation: {stdev}")
        print(f"OpenAI Coefficient: {oai_coeff}")

        print("generating new shapes now")
        # Then, sampling some new shapes -> outputting and rendering them
        x_0s = self.diff.ddim_sample_loop(
            self.model, (16, *self.image_size[1:]), clip_denoised=False
        )
        x_0s = x_0s / self.cfg.normalization_factor

        print(
            "x_0s[0].stats",
            x_0s.min().item(),
            x_0s.max().item(),
            x_0s.mean().item(),
            x_0s.std().item(),
        )
        out_pc_imgs = []
        # Handle 3D generation
        out_imgs = []
        os.makedirs(f"{self.cfg.output_dir}/gen_meshes/{batch_idx}", exist_ok=True)
        self.sample_dir = f"{self.cfg.output_dir}/gen_meshes/{batch_idx}"
        # time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        for x_0 in tqdm(x_0s):
            mesh, _ = self.generate_meshes(x_0.unsqueeze(0), None, res=128)
            mesh = mesh[0]
            if len(mesh.vertices) == 0:
                continue
            mesh.vertices -= mesh.bounding_box.centroid
            mesh.vertices *= 2 / mesh.bounding_box.extents.max()

            name = f"mesh_{len(out_imgs)}_step_{self.global_step}.ply"
            mesh.export(f"{self.sample_dir}/mesh_{len(out_imgs)}.ply")
            print(f"Exported meshes to: {self.sample_dir}/mesh_{len(out_imgs)}.ply")
            render_path = Path(self.sample_dir) / "generated_renders"
            render_path.mkdir(exist_ok=True, parents=True)
            import uuid

            img_name = f"{render_path}/mesh_{len(out_imgs)}_{uuid.uuid4().hex}.png"
            # Scaling the chairs down so that they fit in the camera
            mv_imgs = []
            import pyrender, trimesh, cv2

            scene = pyrender.Scene()
            mesh_node = pyrender.Mesh.from_trimesh(
                mesh,
                material=pyrender.MetallicRoughnessMaterial(
                    alphaMode="BLEND",
                    baseColorFactor=(0 / 255.0, 235 / 255.0, 193 / 255.0),
                    metallicFactor=0.2,
                    roughnessFactor=0.8,
                ),
            )
            scene.add(mesh_node)
            num_viewpoints = 4
            azimuthal_spacing = 2 * np.pi / num_viewpoints
            for view_id in range(4):
                azimuthal_angle = i * azimuthal_spacing
                eye = [
                    np.cos(azimuthal_angle),
                    np.sin(azimuthal_angle),
                    np.tan(np.radians(30)),
                ]
                eye = np.array(eye) / np.sqrt(eye[0] ** 2 + eye[1] ** 2 + eye[2] ** 2)
                # eye *= 2
                target = [0, 0, 0]
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = eye
                scene.add(camera, pose=camera_pose)
                light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)
                scene.add(light, pose=camera_pose)
                r = pyrender.OffscreenRenderer(256, 256)
                img, _ = r.render(scene)
                r.delete()
                import gc

                gc.collect()
                # img, _ = render_mesh(mesh)
                mv_imgs.append(img)
            concatenated_image = np.concatenate(mv_imgs, axis=1)
            from PIL import Image

            cv2.imwrite(
                f"{render_path}/render_{uuid.uuid4().hex}.png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
            import matplotlib.pyplot as plt

            # plt.imsave(concatenated_image, img_name)

            if len(mesh.vertices) > 0:
                pc = torch.tensor(mesh.sample(10000))
            else:
                print("Empty mesh")
                pc = torch.zeros(10000, 3)
            pc_img, _ = render_mesh(pc)
            out_imgs.append(img)
            out_pc_imgs.append(pc_img)

        self.logger.log_image("generated_renders", out_imgs, step=self.current_epoch)
