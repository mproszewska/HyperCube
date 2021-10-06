import urllib
import shutil

from os import listdir, makedirs, remove
from os.path import exists, join
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

from utils.util import load_ply

synth_id_to_category = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02834778": "bicycle",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02858304": "boat",
    "02992529": "cellphone",
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}


class Vox64Dataset(Dataset):
    def __init__(
        self,
        root_dir="/home/datasets/vox64",
        classes=[],
        test_classes=None,
        split="train",
        n_pixels=16,
    ):
        self.root_dir = root_dir
        self.split = split
        self.n_pixels = n_pixels
        self.input_size = 64
        tcf = test_classes[0]
        print(f"classes {classes} | test_classes {test_classes}")
        if len(classes) > 0:
            classes = [category_to_synth_id[c] for c in classes]
            all_classes = False
        else:
            classes = synth_id_to_category.keys()
            all_classes = True

        if test_classes is None:
            test_classes = classes
        else:
            if len(test_classes) > 0:
                test_classes = [category_to_synth_id[c] for c in test_classes]
            else:
                test_classes = synth_id_to_category.keys()

        with h5py.File(f"{root_dir}/all_vox256_img_train.hdf5", "r") as f:
            self.train_points = (
                np.array(f[f"points_{self.n_pixels}"]).astype(np.float32) + 0.5
            ) / 256 - 0.5
            self.train_values = np.array(f[f"values_{self.n_pixels}"])
            self.train_voxels = np.array(f["voxels"]).reshape(
                [-1, 1] + [self.input_size] * 3
            )

        if not all_classes:
            train_include = np.zeros(self.train_points.shape[0], dtype=int)
            with open(f"{root_dir}/all_vox256_img_train.txt") as f:
                for i, l in enumerate(f):
                    for c in classes:
                        if l.startswith(c):
                            train_include[i] = c

            self.train_voxels = self.train_voxels[train_include > 0]
            self.train_points = self.train_points[train_include > 0]
            self.train_values = self.train_values[train_include > 0]

        with h5py.File(f"{root_dir}/all_vox256_img_test.hdf5", "r") as f:
            test_points = (
                np.array(f[f"points_{self.n_pixels}"]).astype(np.float32) + 0.5
            ) / 256 - 0.5
            test_values = np.array(f[f"values_{self.n_pixels}"])
            test_voxels = np.array(f["voxels"]).reshape([-1, 1] + [self.input_size] * 3)

        test_include = np.zeros(test_points.shape[0], dtype=int)
        self.test_names = list()
        with open(f"{root_dir}/all_vox256_img_test.txt") as f:
            for i, l in enumerate(f):
                for c in test_classes:
                    if l.startswith(c):
                        test_include[i] = c
                        self.test_names += [l.strip()]

        self.test_voxels = test_voxels[test_include > 0]
        self.test_points = test_points[test_include > 0]
        self.test_values = test_values[test_include > 0]

        print(f"Objects in train: {self.train_points.shape[0]}")
        print(f"Objects in test: {self.test_points.shape[0]}")

    def __len__(self):
        if self.split == "train":
            return self.train_points.shape[0]
        elif self.split == "valid" or self.split == "test":
            return self.test_points.shape[0]
        else:
            raise ValueError("Invalid split. Should be train, valid or test.")

    def __getitem__(self, idx):
        if self.split == "train":
            x = (self.train_voxels[idx], self.train_points[idx], self.train_values[idx])
        elif self.split == "valid" or self.split == "test":
            x = ((self.test_voxels[idx], self.test_points[idx], self.test_values[idx]),)
        else:
            raise ValueError("Invalid split. Should be train, valid or test.")

        return x

    def get_cloud(self, name):
        return torch.tensor(load_ply(f"{self.root_dir}/../shapenet/{name}.ply"))


def get_coords(multiplier, test_size, frame_grid_size, aux_x, aux_y, aux_z):
    coords = np.zeros([multiplier ** 3, test_size, test_size, test_size, 3], np.float32)
    for i in range(multiplier):
        for j in range(multiplier):
            for k in range(multiplier):
                coords[i * multiplier * multiplier + j * multiplier + k, :, :, :, 0] = (
                    aux_x + i
                )
                coords[i * multiplier * multiplier + j * multiplier + k, :, :, :, 1] = (
                    aux_y + j
                )
                coords[i * multiplier * multiplier + j * multiplier + k, :, :, :, 2] = (
                    aux_z + k
                )
    coords = (coords + 0.5) / frame_grid_size - 0.5
    coords = coords.reshape([multiplier ** 3, test_size ** 3, 3])
    coords = torch.from_numpy(coords)
    return coords


def get_aux(test_size, multiplier):
    aux_x = np.zeros([test_size] * 3, np.uint8)
    aux_y = np.zeros([test_size] * 3, np.uint8)
    aux_z = np.zeros([test_size] * 3, np.uint8)

    for i in range(test_size):
        for j in range(test_size):
            for k in range(test_size):
                aux_x[i, j, k] = i * multiplier
                aux_y[i, j, k] = j * multiplier
                aux_z[i, j, k] = k * multiplier
    return aux_x, aux_y, aux_z
