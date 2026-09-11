# Data loading for the Corr-NMRF WHU remote-sensing stereo release.

import copy
import logging
import os
import os.path as osp
import re
from glob import glob

import numpy as np
import torch
import torch.utils.data as data

from corr_nmrf.utils import dist_utils as comm
from corr_nmrf.utils import evaluation
from corr_nmrf.utils import frame_utils
from corr_nmrf.utils import misc
from .transforms import SparseFlowAugmentor


def read_all_lines(filename):
    with open(filename) as fp:
        return [line.rstrip() for line in fp.readlines()]


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = SparseFlowAugmentor(**aug_params)

        self.disparity_reader = frame_utils.read_gen if reader is None else reader
        self.is_test = False
        self.init_seed = False
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        if self.is_test:
            img1 = np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8)[..., :3]
            img2 = np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8)[..., :3]
            return {
                "img1": torch.from_numpy(img1).permute(2, 0, 1).float(),
                "img2": torch.from_numpy(img2).permute(2, 0, 1).float(),
                "meta": self.extra_info[index],
            }

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            initial_seed = torch.initial_seed() % 2**31
            if worker_info is not None:
                misc.seed_all_rng(initial_seed + worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8)
        img2 = np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        sample = {
            "img1": torch.from_numpy(img1).permute(2, 0, 1).float(),
            "img2": torch.from_numpy(img2).permute(2, 0, 1).float(),
            "disp": torch.from_numpy(flow).permute(2, 0, 1).float()[0],
            "valid": torch.from_numpy(valid),
        }
        return sample

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class WHUStereo(StereoDataset):
    """WHU Stereo loader for the flat RSMT layout and scene-folder layout."""

    _LEFT_NAME_RE = re.compile(r"^(.+)_left_(.+)\.(tif|tiff|png)$", re.I)

    def __init__(self, aug_params=None, root="datasets/WHU_stereo", split="train", index_file=None):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispWHU)
        assert osp.isdir(root), f"WHU root not found: {root}"
        assert split in ("train", "test"), f"WHUStereo split must be train or test, got {split}"

        subset = "train" if split == "train" else "test"
        flat_lr = osp.join(root, subset, "left_right")
        if osp.isdir(flat_lr):
            self._load_flat_layout(root, subset)
            layout = "flat"
        else:
            self._load_scene_layout(root, subset, index_file)
            layout = "scene"
        logging.info("WHUStereo split=%s layout=%s: %d pairs", split, layout, len(self.image_list))

    @staticmethod
    def _find_disparity(disp_dir, prefix, idx):
        for ext in ("tif", "tiff", "png"):
            cand = osp.join(disp_dir, f"{prefix}_disparity_{idx}.{ext}")
            if osp.isfile(cand):
                return cand
        return None

    def _load_flat_layout(self, root, subset):
        lr_dir = osp.join(root, subset, "left_right")
        disp_dir = osp.join(root, subset, "disparity")
        assert osp.isdir(disp_dir), f"WHU flat layout missing disparity dir: {disp_dir}"
        for lf in sorted(glob(osp.join(lr_dir, "*"))):
            match = self._LEFT_NAME_RE.match(osp.basename(lf))
            if match is None:
                continue
            prefix, idx, ext = match.group(1), match.group(2), match.group(3)
            rf = osp.join(lr_dir, f"{prefix}_right_{idx}.{ext}")
            df = self._find_disparity(disp_dir, prefix, idx)
            if osp.isfile(rf) and df is not None:
                self.image_list.append([lf, rf])
                self.disparity_list.append(df)

    def _load_scene_layout(self, root, subset, index_file):
        idx_path = osp.join(root, index_file or f"{subset}_index.txt")
        if osp.isfile(idx_path):
            scenes = [ln.strip() for ln in read_all_lines(idx_path) if ln.strip() and not ln.startswith("#")]
        else:
            subset_root = osp.join(root, subset)
            scenes = sorted(d for d in os.listdir(subset_root) if osp.isdir(osp.join(subset_root, d)))
            logging.warning("WHUStereo: index file %s not found, using all scenes under %s", idx_path, subset)

        for scene in scenes:
            scene_root = osp.join(root, subset, scene)
            left_dir = osp.join(scene_root, "Left")
            if not osp.isdir(left_dir):
                continue
            for lf in sorted(glob(osp.join(left_dir, "*.png"))):
                base = osp.basename(lf)
                rf = osp.join(scene_root, "Right", base)
                df = osp.join(scene_root, "Disparity", base)
                if osp.isfile(rf) and osp.isfile(df):
                    self.image_list.append([lf, rf])
                    self.disparity_list.append(df)


def build_train_loader(cfg):
    crop_size = cfg.DATASETS.CROP_SIZE
    spatial_scale = cfg.DATASETS.SPATIAL_SCALE
    aug_params = {
        "crop_size": list(crop_size),
        "min_scale": spatial_scale[0],
        "max_scale": spatial_scale[1],
        "do_flip": cfg.DATASETS.DO_FLIP if cfg.DATASETS.DO_FLIP is not None else False,
        "yjitter": cfg.DATASETS.YJITTER,
        "random_crop": getattr(cfg.DATASETS, "WHU_RANDOM_CROP", getattr(cfg.DATASETS, "RANDOM_CROP", True)),
        "spatial_crop": getattr(cfg.DATASETS, "WHU_SPATIAL_CROP", getattr(cfg.DATASETS, "SPATIAL_CROP", True)),
    }
    if cfg.DATASETS.SATURATION_RANGE is not None:
        aug_params["saturation_range"] = cfg.DATASETS.SATURATION_RANGE
    if cfg.DATASETS.IMG_GAMMA is not None:
        aug_params["gamma"] = cfg.DATASETS.IMG_GAMMA

    train_dataset = None
    logger = logging.getLogger(__name__)
    for dataset_name in cfg.DATASETS.TRAIN:
        if dataset_name != "whu_stereo":
            raise ValueError(f"This Corr-NMRF release only supports whu_stereo training, got {dataset_name}")
        whu_root = getattr(cfg.DATASETS, "WHU_ROOT", "") or ""
        assert whu_root, "DATASETS.WHU_ROOT must be set for whu_stereo"
        new_dataset = WHUStereo(dict(aug_params), root=whu_root, split="train")
        logger.info("Adding %d samples from WHU Stereo train", len(new_dataset))
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    world_size = comm.get_world_size()
    total_batch_size = cfg.SOLVER.IMS_PER_BATCH
    assert total_batch_size > 0 and total_batch_size % world_size == 0
    batch_size = total_batch_size // world_size

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=comm.get_world_size(), rank=comm.get_rank()
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    return train_loader, train_sampler


def build_val_loader(cfg, dataset_name):
    if dataset_name != "whu_stereo_test":
        raise ValueError(f"This Corr-NMRF release only supports whu_stereo_test validation, got {dataset_name}")

    whu_root = getattr(cfg.DATASETS, "WHU_ROOT", "") or ""
    assert whu_root, "DATASETS.WHU_ROOT must be set for whu_stereo_test"
    val_dataset = WHUStereo(aug_params=None, root=whu_root, split="test")
    logging.getLogger(__name__).info("Number of validation image pairs (WHU test): %d", len(val_dataset))

    val_sampler = evaluation.InferenceSampler(len(val_dataset)) if comm.get_world_size() > 1 else None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
    )
    return val_loader
