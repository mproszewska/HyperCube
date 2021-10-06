import argparse
from itertools import chain
import json
import logging
import numpy as np
import os
from os.path import join, exists
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.vox64 import *
from datasets.shapenet import *
from losses.champfer_loss import ChamferLoss
from models import implicit_ae, aae
from utils.pcutil import plot_3d_point_cloud
from utils.points import generate_points
from utils.ply_utils import *
from utils.util import *

import mcubes
import open3d as o3d

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def load_dataset(config, n_pixels):
    dataset_name = "vox64"
    dataset = Vox64Dataset(
        root_dir=config["data_dir"],
        classes=config["classes"],
        test_classes=config["test_classes"],
        n_pixels=n_pixels,
    )

    log.info(
        "Selected {} classes. Loaded {} samples.".format(
            "all" if not config["classes"] else ",".join(config["classes"]),
            len(dataset),
        )
    )

    points_dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    return dataset, points_dataloader


log = logging.getLogger("HyperCube")


def main(config):
    # Preparing shapenet dataset for calculating Chamfer distance
    dataset = ShapeNetDataset(
        root_dir=config["shapenet_dir"],
        classes=["airplane", "car", "chair", "rifle", "table"],
    )
    
    config["dataset"] = "vox64"
    set_seed(config["seed"])
    model_type = "hypercube"

    if config["intervals"]:
        name = "hypercube_interval"
    else:
        name = "hypercube"

    results_dir = prepare_results_dir(config, name, "training")

    starting_epoch = find_latest_epoch(results_dir) + 1
    print(results_dir, "starting epoch", starting_epoch)

    if not exists(join(results_dir, "config.json")):
        with open(join(results_dir, "config.json"), mode="w") as f:
            json.dump(config, f)

    setup_logging(results_dir)

    device = cuda_setup(config["cuda"], config["gpu"])
    log.info(f"Device variable: {device}")
    if device.type == "cuda":
        log.info(f"Current CUDA device: {torch.cuda.current_device()}")

    weights_path = join(results_dir, "weights")

    #
    # Dataset
    #
    n_pixels = config["n_pixels"]
    n_pixels_epoch = config["n_pixels_epoch"]
    n_pixels = 16 * (2 ** (starting_epoch // n_pixels_epoch))
    n_pixels = min(64, n_pixels)
    dataset, points_dataloader = load_dataset(config, n_pixels)

    #
    # Models
    generator = aae.HyperNetwork(config, device).to(device)
    generator.apply(weights_init)

    encoder = implicit_ae.Encoder(config).to(device)

    def network_loss(G, point_value):
        return torch.sum((G - point_value) ** 2)

    reconstruction_loss = network_loss
    spec_loss = torch.nn.CrossEntropyLoss()

    #
    # Optimizers
    #
    e_hn_optimizer = torch.optim.Adam(
        chain(encoder.parameters(), generator.parameters()),
        **config["optimizer"]["E_HN"]["hyperparams"],
    )

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading weights...")
        epoch = starting_epoch - 1
        generator.load_state_dict(torch.load(join(weights_path, f"{epoch:05}_G.pth")))
        encoder.load_state_dict(torch.load(join(weights_path, f"{epoch:05}_E.pth")))

        e_hn_optimizer.load_state_dict(
            torch.load(join(weights_path, f"{epoch:05}_EGo.pth"))
        )

        log.info("Loading losses...")
    else:
        log.info("First epoch")
        losses_all = []

    target_network_input = None

    for epoch in range(starting_epoch, config["max_epochs"] + 1):
        log.debug("Epoch: %s" % epoch)
        generator.train()
        encoder.train()

        total_loss = 0.0
        total_rec, total_kld, total_spec, count = 0, 0, 0, 0
        if n_pixels != 64:
            if n_pixels_epoch > 0 and epoch == n_pixels_epoch + 1:
                n_pixels = 32
                del dataset, points_dataloader
                dataset, points_dataloader = load_dataset(config, n_pixels)
            if n_pixels_epoch > 0 and epoch == 2 * n_pixels_epoch + 1:
                n_pixels = 64
                del dataset, points_dataloader
                dataset, points_dataloader = load_dataset(config, n_pixels)

        for i, point_data in enumerate(points_dataloader, 1):
            (voxels, point_coord, point_value) = point_data
            voxels, point_coord, point_value = (
                voxels.to(device).float(),
                point_coord.to(device).float(),
                point_value.to(device).float(),
            )
            if n_pixels == 64:
                batch = np.random.randint(4)
                size = 16 * 16 * 16
                point_coord = point_coord[:, batch * size : (batch + 1) * size]
                point_value = point_value[:, batch * size : (batch + 1) * size]

            e_hn_optimizer.zero_grad()

            z = encoder(voxels)
            eps_val = 0
            target_networks_weights = generator(z)
            shape = (
                (point_value.shape[0], point_value.shape[1], 2)
                if config["intervals"]
                else point_value.shape
            )
            net_out = torch.zeros(shape).to(device)
            z_out = torch.zeros(shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(
                    device
                )
                if config["intervals"]:
                    eps_val = min(epoch * 0.000002, 0.001)
                    eps = eps_val * torch.ones_like(point_coord[j]).to(device)
                    pred, eps, z_l, z_u = target_network.forward_with_eps(
                        point_coord[j], eps
                    )
                    net_out[j] = pred
                    tmp = F.one_hot(point_value[j, 0].long(), pred.size(-1))
                    z_out[j] = torch.where(tmp.bool(), z_l, z_u)
                else:
                    net_out[j] = target_network(point_coord[j])

            if config["intervals"]:
                point_value_flatten = point_value.reshape(-1).long()
                loss_rec = F.cross_entropy(net_out.reshape(-1, 2), point_value_flatten, reduction="sum")
                loss_spec = F.cross_entropy(z_out.reshape(-1, 2), point_value_flatten, reduction="sum")
                kappa = max(1 - (epoch * 0.001), 0.5)
            else:
                loss_spec = torch.tensor([0]).to(device)
                kappa = 1
                loss_rec = reconstruction_loss(net_out, point_value)
            loss = kappa * loss_rec + (1 - kappa) * loss_spec

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                chain(encoder.parameters(), generator.parameters()), 1.0
            )
            e_hn_optimizer.step()

            total_loss += loss.item()
            total_rec += loss_rec.item()
            total_spec += loss_spec.item()
            count += 1
        print(
            f"kappa: {kappa:.4f} | eps: {eps_val:.4f} | rec: {total_rec/count:.4f} | spec: {total_spec/count:.4f}"
        )

        #
        # Save intermediate results
        #

        if epoch % config["save_samples_frequency"] == 0:
            log.debug("Saving samples...")
            encoder.eval()
            generator.eval()
            with torch.no_grad():
                sample(
                    model_type,
                    results_dir,
                    device,
                    dataset,
                    encoder,
                    generator,
                    config,
                    "train_" + str(n_pixels) + "_" + str(epoch),
                    n=10,
                )
        if config["clean_weights_dir"]:
            log.debug("Cleaning weights path: %s" % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config["save_weights_frequency"] == 0:
            log.debug("Saving weights and losses...")

            torch.save(generator.state_dict(), join(weights_path, f"{epoch:05}_G.pth"))
            torch.save(encoder.state_dict(), join(weights_path, f"{epoch:05}_E.pth"))
            torch.save(
                e_hn_optimizer.state_dict(), join(weights_path, f"{epoch:05}_EGo.pth")
            )

        if epoch % config["calculate_metrics_frequency"] == 0:
            log.debug("Calculating metrics...")
            encoder.eval()
            generator.eval()
            with torch.no_grad():
                sample(
                    model_type,
                    results_dir,
                    device,
                    dataset,
                    encoder,
                    generator,
                    config,
                    "",
                    n=100,
                    calculate_metrics=True,
                    save_images=False,
                )


def save_latent(config, encoder, points_dataloader, dataset, results_dir, device):
    print("Saving latent")
    frame_grid_size = 64
    test_size = 32
    multiplier = int(frame_grid_size / test_size)

    point_coord = torch.zeros(1, 32 * 32 * 32, 3, device=device)
    zs = torch.zeros(len(dataset.train_voxels), config["z_size"])
    idx = 0
    for point_data in points_dataloader:
        (voxels, _, _) = point_data
        print(voxels.shape)
        voxels = voxels.to(device).float()
        z = encoder(voxels)
        if isinstance(z, tuple):
            z = z[0]
        zs[idx : idx + len(z)] = z
        idx += len(z)
    torch.save(zs, f"{results_dir}/latent.pt")
    print(f"Latent saved to {results_dir}/latent.pt")


def sample(
    model_type,
    results_dir,
    device,
    dataset,
    encoder,
    generator,
    config,
    name,
    n=10,
    calculate_metrics=False,
    save_images=True,
    frame=64,
    eps_val=0,
):

    frame_grid_size = frame
    test_size = 32
    multiplier = int(frame_grid_size / test_size)
    point_coord = torch.zeros(1, 32 * 32 * 32, 3, device=device)

    batch_voxels = torch.from_numpy(dataset.test_voxels[:n]).to(device).float()
    z = encoder(batch_voxels)
    model_float = np.zeros([z.shape[0]] + [frame_grid_size + 2] * 3, np.float32)

    aux_x, aux_y, aux_z = get_aux(test_size, multiplier)
    coords = get_coords(multiplier, test_size, frame_grid_size, aux_x, aux_y, aux_z).to(
        device
    )

    for i in range(multiplier):
        for j in range(multiplier):
            for k in range(multiplier):
                minib = i * multiplier * multiplier + j * multiplier + k
                point_coord = coords[minib : minib + 1]
                target_networks_weights = generator(z)
                net_out = torch.zeros(z.shape[0], 32 * 32 * 32, 1).to(device)
                for t, target_network_weights in enumerate(target_networks_weights):
                    target_network = aae.TargetNetwork(
                        config, target_network_weights
                    ).to(device)
                    if config["intervals"]:
                        eps = eps_val * torch.ones_like(point_coord[0])
                        pred = target_network.forward_with_eps(point_coord[0], eps)[0]
                        pred = F.softmax(pred, dim=1)
                        net_out[t] = pred[:, 1:2]
                    else:
                        net_out[t] = target_network(point_coord[0])

                model_float[
                    :, aux_x + i + 1, aux_y + j + 1, aux_z + k + 1
                ] = np.reshape(net_out.detach().cpu().numpy(), [-1] + [test_size] * 3)
    chamfer = 0
    coords = coords.reshape(frame_grid_size, frame_grid_size, frame_grid_size, 3)
    for i in range(z.shape[0]):

        chamfer_loss = ChamferLoss().to(device)
        vertices, triangles = mcubes.marching_cubes(model_float[i], 0.5)
        if len(vertices) == 0:
            continue
        vertices = (vertices.astype(np.float32) / frame_grid_size) - 0.5
        filename = f"{results_dir}/samples/{name}_{i}"
        if save_images:
            write_ply_triangle(f"{filename}.ply", vertices, triangles)
        if not calculate_metrics:
            continue
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
        )
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=2048)
        sampled_points = torch.tensor(np.asarray(pcd.points)).float().to(device)
        points = dataset.get_cloud(dataset.test_names[i]).to(device)

        batch_voxels_bis = torch.zeros((1, 66, 66, 66, 1))
        batch_voxels_bis[0, 1:-1, 1:-1, 1:-1, 0] = batch_voxels[i]
        vertices, triangles = mcubes.marching_cubes(
            batch_voxels_bis[0, :, :, :, 0].cpu().detach().numpy(), 0.5
        )
        original_points = vertices = (
            (vertices.astype(np.float32) / frame_grid_size) - 0.5
        ).astype(np.float32)
        ch = chamfer_loss(
            sampled_points.unsqueeze(0).cuda(),
            torch.from_numpy(original_points).unsqueeze(0).cuda(),
            mean=True,
        )
        chamfer += ch

    if calculate_metrics:
        torch.set_printoptions(precision=5, sci_mode=False)
        model_float = torch.tensor(model_float[:, 1:-1, 1:-1, 1:-1]).float().to(device)

        model_float = model_float > 0.5
        mse = (model_float.float() - batch_voxels[:, 0]) ** 2
        batch_voxels = batch_voxels.bool()

        iou = 0
        for i in range(n):
            p = torch.sum(model_float[i] & batch_voxels[i])
            q = torch.sum(model_float[i] | batch_voxels[i])
            iou += float(p) / float(q)
        print(f"MSE: {torch.mean(mse):.6f} | IoU {iou/n:.6f} | Chamfer {chamfer/n:.6f}")


if __name__ == "__main__":
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, type=str, help="config file path"
    )
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith(".json"):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
