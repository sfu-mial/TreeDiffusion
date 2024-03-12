# Source: https://github.com/luost26/diffusion-point-cloud/blob/main/evaluation/evaluation_metrics.py

"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""
import os
import warnings

import numpy as np
import torch
import trimesh
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

import wandb


def emd_approx(sample, ref):
    emd_val = torch.zeros([sample.size(0)]).to(sample)
    return emd_val


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(ref_pcs, batch_size, diff_module, n_points, reduced=True):
    N_ref = ref_pcs.shape[0]

    cd_lst = []
    iterator = range(0, N_ref, batch_size)

    for b_start in tqdm(iterator, desc="EMD-CD"):
        b_end = min(N_ref, b_start + batch_size)
        ref_batch = ref_pcs[b_start:b_end]
        sample_x_0s = diff_module.diff.sample(len(ref_batch))
        sample_meshes, _ = diff_module.generate_meshes(sample_x_0s, None)
        sample_batch = []
        for mesh in sample_meshes:
            pc = torch.tensor(mesh.sample(n_points))
            sample_batch.append(pc)
        sample_batch = torch.stack(sample_batch)
        # sample_batch = torch.randn(*ref_batch.shape).double()
        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

    if reduced:
        cd = torch.cat(cd_lst).mean()
    else:
        cd = torch.cat(cd_lst)

    results = {
        "MMD-CD": cd,
    }
    return results


def _pairwise_EMD_CD_4D(sample_pcs, ref_pcs, batch_size, verbose=True):
    M_rs_cd_total, M_rs_emd_total = None, None
    for t in range(sample_pcs.shape[1]):
        M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(
            sample_pcs[:, t, ...], ref_pcs[:, t, ...], batch_size, verbose=verbose
        )
        if t == 0:
            M_rs_cd_total, M_rs_emd_total = M_rs_cd, M_rs_emd
        else:
            M_rs_cd_total += M_rs_cd
            M_rs_emd_total += M_rs_emd
    M_rs_cd_total /= sample_pcs.shape[1]
    M_rs_emd_total /= sample_pcs.shape[1]
    return M_rs_cd_total, M_rs_emd_total


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc="Pairwise EMD-CD")
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1
            )
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat(
        [torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0
    )
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False
    )

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


def lgan_mmd_cov_match(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }, min_idx.view(-1)


def compute_all_metrics_4d(sample_pcs, ref_pcs, batch_size, logger):
    results = {}

    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_4D(ref_pcs, sample_pcs, batch_size)

    ## EMD
    res_emd = lgan_mmd_cov(M_rs_emd.t())
    # results.update({
    #     "%s-EMD" % k: v for k, v in res_emd.items()
    # })

    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())

    # We use the below code to visualize some goodly&badly performing shapes
    # you can uncomment if you want to analyze that

    # r = Rotation.from_euler('x', 90, degrees=True)
    # min_dist, min_dist_sample_idx = torch.min(M_rs_cd.t(), dim=0)
    # min_dist_sorted_idx = torch.argsort(min_dist)
    #
    # orig_meshes_dir = f"orig_meshes/run_{wandb.run.name}"
    # os.makedirs(orig_meshes_dir, exist_ok=True)
    # for i, ref_id in enumerate(min_dist_sorted_idx):
    #     ref_id = ref_id.item()
    #     matched_sample_id = min_dist_sample_idx[ref_id].item()
    #
    #     for time in range(16):
    #         mlp_pc = trimesh.points.PointCloud(sample_pcs[matched_sample_id, time].cpu())
    #         mlp_pc.export(f"{orig_meshes_dir}/mlp_top_{i}_{time}.obj")
    #         mlp_pc = trimesh.points.PointCloud(ref_pcs[ref_id, time].cpu())
    #         mlp_pc.export(f"{orig_meshes_dir}/mlp_top_{i}_{time}_ref.obj")
    # #
    # # for i, ref_id in enumerate(min_dist_sorted_idx[:4]):
    # #     ref_id = ref_id.item()
    # #     matched_sample_id = min_dist_sample_idx[ref_id].item()
    # #     logger.experiment.log({f'pc/top_{i}': [
    # #         wandb.Object3D(r.apply(sample_pcs[matched_sample_id].cpu())),
    # #         wandb.Object3D(r.apply(ref_pcs[ref_id].cpu()))]})
    # # for i, ref_id in enumerate(reversed(min_dist_sorted_idx[-4:])):
    # #     ref_id = ref_id.item()
    # #     matched_sample_id = min_dist_sample_idx[ref_id].item()
    # #     logger.experiment.log({f'pc/bottom_{i}': [
    # #         wandb.Object3D(r.apply(sample_pcs[matched_sample_id].cpu())),
    # #         wandb.Object3D(r.apply(ref_pcs[ref_id].cpu()))]})
    # print(min_dist, min_dist_sample_idx, min_dist_sorted_idx)
    # print("Sorted:", min_dist[min_dist_sorted_idx])

    results.update({"%s-CD" % k: v for k, v in res_cd.items()})
    for k, v in results.items():
        print("[%s] %.8f" % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_4D(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_4D(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update(
        {"1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if "acc" in k}
    )

    ## EMD
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    # results.update({
    #     "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    # })

    return results


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, logger):
    results = {}
    logger, orig_meshes_dir = logger

    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    ## EMD
    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({"%s-EMD" % k: v for k, v in res_emd.items()})

    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())

    # We use the below code to visualize some goodly&badly performing shapes
    # you can uncomment if you want to analyze that

    if len(sample_pcs) > 5:
        r = Rotation.from_euler("x", 90, degrees=True)
        min_dist, min_dist_sample_idx = torch.min(M_rs_cd.t(), dim=0)
        min_dist_sorted_idx = torch.argsort(min_dist)
        # orig_meshes_dir = f"orig_meshes/run_{wandb.run.name}"

        for i, ref_id in enumerate(min_dist_sorted_idx):
            ref_id = ref_id.item()
            matched_sample_id = min_dist_sample_idx[ref_id].item()
            mlp_pc = trimesh.points.PointCloud(sample_pcs[matched_sample_id].cpu())
            mlp_pc.export(f"{orig_meshes_dir}/mlp_top{i}.obj")
            mlp_pc = trimesh.points.PointCloud(ref_pcs[ref_id].cpu())
            mlp_pc.export(f"{orig_meshes_dir}/mlp_top{i}ref.obj")

        for i, ref_id in enumerate(min_dist_sorted_idx[:4]):
            ref_id = ref_id.item()
            matched_sample_id = min_dist_sample_idx[ref_id].item()
            logger.experiment.log(
                {
                    f"pc/top_{i}": [
                        wandb.Object3D(r.apply(sample_pcs[matched_sample_id].cpu())),
                        wandb.Object3D(r.apply(ref_pcs[ref_id].cpu())),
                    ]
                }
            )
        for i, ref_id in enumerate(reversed(min_dist_sorted_idx[-4:])):
            ref_id = ref_id.item()
            matched_sample_id = min_dist_sample_idx[ref_id].item()
            logger.experiment.log(
                {
                    f"pc/bottom_{i}": [
                        wandb.Object3D(r.apply(sample_pcs[matched_sample_id].cpu())),
                        wandb.Object3D(r.apply(ref_pcs[ref_id].cpu())),
                    ]
                }
            )
        print(min_dist, min_dist_sample_idx, min_dist_sorted_idx)
        torch.save(min_dist, f"{orig_meshes_dir}/min_dist.pth")
        torch.save(min_dist_sample_idx, f"{orig_meshes_dir}/min_dist_sample_idx.pth")
        torch.save(min_dist_sorted_idx, f"{orig_meshes_dir}/min_dist_sorted_idx.pth")
        torch.save(
            min_dist[min_dist_sorted_idx], f"{orig_meshes_dir}/min_dist_sorted.pth"
        )
        print("Sorted:", min_dist[min_dist_sorted_idx])

    results.update({"%s-CD" % k: v for k, v in res_cd.items()})
    for k, v in results.items():
        print("[%s] %.8f" % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update(
        {"1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if "acc" in k}
    )

    ## EMD
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    results.update(
        {"1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if "acc" in k}
    )

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[
        1
    ]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn("Point-clouds are not in unit cube.")

    if in_sphere and np.max(np.sqrt(np.sum(pclouds**2, axis=2))) > bound:
        if verbose:
            warnings.warn("Point-clouds are not in unit sphere.")

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc="JSD"):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError("Negative values.")
    if len(P) != len(Q):
        raise ValueError("Non equal size.")

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn("Numerical values of two JSD methods don't agree.")

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


if __name__ == "__main__":
    import os
    from glob import glob
    from pathlib import Path
    import sys

    dataroot = "/local-scratch2/asa409/data/brain_vessels/IntrA/complete/"
    intra_meshes = ["AN121", "AN116", "AN137", "AN9", "AN168", "AN6", "AN55", "AN58"]
    all_intra_meshes = sorted(list((Path(dataroot)).glob("*_fixed.obj")))
    all_intra_meshes = [
        x
        for x in all_intra_meshes
        if str(x).split("/")[-1].split("_")[0].split("ArteryObj")[-1] in intra_meshes
    ]
    pred_meshes_5x512 = glob(
        "/local-scratch2/asa409/data/brain_vessels/IntrA/complete/pred_meshes/pred_mlp_512/*.ply"
    )
    pred_meshes_5x128 = glob(
        "/local-scratch2/asa409/data/brain_vessels/IntrA/complete/pred_meshes/pred_mlp_128/*.ply"
    )
    pred_meshes_5x64 = glob(
        "/local-scratch2/asa409/data/brain_vessels/IntrA/complete/pred_meshes/pred_mlp_64/*.ply"
    )
    pred_meshes_5x256 = glob(
        "/local-scratch2/asa409/data/brain_vessels/IntrA/complete/pred_meshes/pred_mlp_256/*.ply"
    )
    import numpy as np
    import trimesh

    # from scipy.spatial.kdtree import KDTree
    from scipy.spatial import cKDTree as KDTree

    def normalize_to_unit_cube(mesh):
        scale_factor = 0.8
        center = mesh.bounding_box.centroid
        length = np.max(mesh.bounding_box.extents)
        scale = 2.0 / length
        scale *= scale_factor
        mesh.vertices -= center
        mesh.vertices *= scale
        return mesh

    from itertools import product
    from tqdm import tqdm
    import pandas as pd

    df = pd.DataFrame()
    layer_series = []
    dim_series = []
    mc_res_series = []
    gt_size_series = []
    pred_size_series = []
    gt_verts_series = []
    pred_verts_series = []

    # metrics
    eval_dicts = []

    layers = [1, 3, 5]
    dims = [64, 128, 256, 512, 1024]
    pred_root_intra = (
        "/local-scratch2/asa409/data/brain_vessels/IntrA/complete/pred_meshes/"
    )
    from eval_metrics import (
        MeshEvaluator,
        compute_iou,
        check_mesh_contains,
        distance_p2m,
        distance_p2p,
    )

    for name in tqdm(all_intra_meshes):
        mesh_name = name.stem.split("_")[0]
        for layer, dim in tqdm(product(layers, dims)):
            for mc_res in tqdm([32, 64, 128, 256]):
                pred_meshes_path = (
                    Path(pred_root_intra)
                    / f"pred_mlp_{dim}"
                    / f"{mesh_name}_pred_{mc_res}.ply"
                )
                if not pred_meshes_path.exists():
                    continue
                gt_mesh = trimesh.load_mesh(name)
                gt_mesh = normalize_to_unit_cube(gt_mesh)
                pred_mesh = trimesh.load_mesh(pred_meshes_path)
                pred_mesh = normalize_to_unit_cube(pred_mesh)
                gt_pc, gt_idx = gt_mesh.sample(50000, return_index=True)
                pred_pc, pred_idx = pred_mesh.sample(50000, return_index=True)
                gt_normals = gt_mesh.face_normals[gt_idx]
                pred_normals = pred_mesh.face_normals[pred_idx]

                layer_series.append(layer)
                dim_series.append(dim)
                mc_res_series.append(mc_res)
                gt_size = os.stat(name).st_size / 1024**2
                pred_size = os.stat(pred_meshes_path).st_size / 1024**2
                gt_size_series.append(gt_size)
                pred_size_series.append(pred_size)
                gt_verts_series.append(len(gt_mesh.vertices))
                pred_verts_series.append(len(pred_mesh.vertices))

                evaluator = MeshEvaluator(50000)
                pc_metrics = evaluator.eval_pointcloud(
                    pred_pc, gt_pc, pred_normals, gt_normals
                )
                tree = KDTree(pred_pc)
                dist, _ = tree.query(gt_pc)
                d1 = dist
                gt_to_gen_chamfer = np.mean(dist)
                gt_to_gen_chamfer_sq = np.mean(np.square(dist))

                tree = KDTree(gt_pc)
                dist, _ = tree.query(pred_pc)
                d2 = dist
                gen_to_gt_chamfer = np.mean(dist)
                gen_to_gt_chamfer_sq = np.mean(np.square(dist))

                cd = gt_to_gen_chamfer + gen_to_gt_chamfer
                # metrics = dict()
                pc_metrics["cd"] = cd
                th = [0.01, 0.02, 0.05]

                if len(d1) and len(d2):
                    for thres in th:
                        recall = float(sum(d < thres for d in d2)) / float(len(d2))
                        precision = float(sum(d < thres for d in d1)) / float(len(d1))

                        if recall + precision > 0:
                            fscore = 2 * recall * precision / (recall + precision)
                        else:
                            fscore = 0
                        pc_metrics[f"recall@{thres}"] = recall
                        pc_metrics[f"precision@{thres}"] = precision
                        pc_metrics[f"fscore@{thres}"] = fscore
                # print(pc_metrics)
                # break
                pc_metrics["mesh_name"] = mesh_name
                pc_metrics["layer"] = layer
                pc_metrics["dim"] = dim
                pc_metrics["mc_res"] = mc_res
                pc_metrics["gt_size"] = gt_size
                pc_metrics["pred_size"] = pred_size
                pc_metrics["gt_verts"] = len(gt_mesh.vertices)
                pc_metrics["pred_verts"] = len(pred_mesh.vertices)
                eval_dicts.append(pc_metrics)
    # df["layer"] = layer_series
    # df["dim"] = dim_series
    # df["mc_res"] = mc_res_series
    # df["gt_size"] = gt_size_series
    # df["pred_size"] = pred_size_series
    # df["gt_verts"] = gt_verts_series
    # df["pred_verts"] = pred_verts_series
    df = pd.DataFrame(eval_dicts)
    df.to_csv("~/DATA/INR_paper_plots/intra_mlp_metrics.csv")

    a = torch.randn([16, 2048, 3]).cuda()
    b = torch.randn([16, 2048, 3]).cuda()
    print(EMD_CD(a, b, batch_size=16))
