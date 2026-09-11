import argparse
import glob
import multiprocessing as mp
import os

import cv2
import torch

from corr_nmrf.config import get_cfg
from corr_nmrf.data import StereoDataset, WHUStereo
from corr_nmrf.models import build_model
from corr_nmrf.utils import visualization
from corr_nmrf.utils.logger import setup_logger


def setup_cfg(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Corr-NMRF inference")
    parser.add_argument("--config-file", default="configs/whu_stereo_resnet.yaml", metavar="FILE")
    parser.add_argument("--dataset-name", choices=["whu_stereo_test"], help="Run on the WHU test split.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Input image pairs, or two glob patterns such as 'left/*.tif right/*.tif'.",
    )
    parser.add_argument("--output", help="Directory to save visualized disparity maps.")
    parser.add_argument("--show-attr", default="disparity", choices=["disparity"])
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    return parser


def _expand_input_pairs(inputs):
    if len(inputs) == 2 and any(ch in inputs[0] + inputs[1] for ch in "*?[]"):
        left = sorted(glob.glob(inputs[0]))
        right = sorted(glob.glob(inputs[1]))
        if len(left) != len(right):
            raise ValueError(f"Input glob counts do not match: {len(left)} left vs {len(right)} right")
        return list(zip(left, right))

    if len(inputs) % 2 != 0:
        raise ValueError("--input expects an even number of files or two glob patterns")
    n_pairs = len(inputs) // 2
    return list(zip(inputs[:n_pairs], inputs[n_pairs:]))


def _find_output_path(root):
    root = os.path.normpath(root)

    def wrapper(file_path):
        file_path = os.path.normpath(file_path)
        try:
            rel = os.path.relpath(file_path, root)
        except ValueError:
            rel = os.path.basename(file_path)
        if rel.startswith(".."):
            rel = os.path.basename(file_path)
        stem, _ = os.path.splitext(rel)
        return stem + ".png"

    return wrapper


@torch.no_grad()
def run_on_dataset(dataset, model, output, find_output_path):
    model.eval()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        rgb = sample["img1"].permute(1, 2, 0).numpy()
        viz = visualization.Visualizer(rgb)

        sample = {"img1": sample["img1"][None], "img2": sample["img2"][None]}
        result_dict = model(sample)
        disp_pred = result_dict["disp"][0].cpu()
        visualized_output = viz.draw_disparity(disp_pred, colormap="kitti")

        if output:
            output_path = os.path.join(output, find_output_path(dataset.image_list[idx][0]))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            visualized_output.save(output_path)
        else:
            cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
            cv2.imshow("disparity", visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="corr_nmrf")
    logger = setup_logger()
    logger.info("Arguments: %s", args)

    cfg = setup_cfg(args)
    model = build_model(cfg)[0].to(torch.device("cuda"))
    checkpoint = torch.load(cfg.SOLVER.RESUME, map_location="cuda")
    weights = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(weights, strict=cfg.SOLVER.STRICT_RESUME)

    if args.dataset_name:
        dataset = WHUStereo(root=cfg.DATASETS.WHU_ROOT, split="test")
        out_root = cfg.DATASETS.WHU_ROOT
    elif args.input:
        image_list = _expand_input_pairs(args.input)
        dataset = StereoDataset()
        dataset.image_list = image_list
        dataset.is_test = True
        dataset.extra_info = [None] * len(image_list)
        out_root = os.path.dirname(image_list[0][0]) or "."
    else:
        raise ValueError("Please provide --dataset-name whu_stereo_test or --input image pairs")

    run_on_dataset(dataset, model, args.output, _find_output_path(out_root))
