#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Export binary Bike Lane masks from OneFormer Mapillary Vistas semantic output.
# Bike Lane class id in Mapillary Vistas: 7
# ------------------------------------------------------------------------------

import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.join(REPO_ROOT, "demo")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, DEMO_DIR)

from defaults import DefaultPredictor  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data.detection_utils import read_image  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from oneformer import (  # noqa: E402
    add_common_config,
    add_convnext_config,
    add_dinat_config,
    add_oneformer_config,
    add_swin_config,
)


MAPILLARY_BIKE_LANE_ID = 7


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.IS_TRAIN = False
    cfg.MODEL.IS_DEMO = True
    cfg.freeze()
    return cfg


def expand_inputs(input_args):
    paths = []
    for item in input_args:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(matches)
        elif os.path.isfile(item):
            paths.append(item)
    return sorted(dict.fromkeys(paths))


def output_path(output_dir, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, f"{stem}.png")


def save_mask(predictions, class_id, out_file):
    sem_map = predictions["sem_seg"].argmax(dim=0).to("cpu").numpy()
    binary_mask = (sem_map == class_id).astype(np.uint8) * 255
    if not cv2.imwrite(out_file, binary_mask):
        raise OSError(f"Failed to write mask: {out_file}")
    return int(binary_mask.sum() // 255)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Export Bike Lane binary masks from OneFormer Mapillary Vistas semantic inference."
    )
    parser.add_argument(
        "--config-file",
        default="configs/mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml",
        help="Path to OneFormer Mapillary Vistas config.",
    )
    parser.add_argument(
        "--weights",
        default="checkpoints/oneformer_mapillary_swin_large.pth",
        help="Path to OneFormer Mapillary Vistas checkpoint.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input images or glob patterns, e.g. 'inputs/RGB_frames/*.jpg'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where binary masks are saved as PNG files.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=MAPILLARY_BIKE_LANE_ID,
        help="Mapillary Vistas class id to export. Default: 7 (Bike Lane).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose output PNG already exists.",
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Additional config overrides using KEY VALUE pairs.",
    )
    return parser


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    image_paths = expand_inputs(args.input)
    if not image_paths:
        raise ValueError("No input images found.")

    os.makedirs(args.output, exist_ok=True)

    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    total_pixels = 0
    processed = 0
    skipped = 0
    start_all = time.time()

    for image_path in tqdm.tqdm(image_paths):
        out_file = output_path(args.output, image_path)
        if args.skip_existing and os.path.exists(out_file):
            skipped += 1
            continue

        image = read_image(image_path, format="BGR")
        # Match demo/predictor.py exactly so binary masks align with semantic_inference visuals.
        image = image[:, :, ::-1]
        predictions = predictor(image, "semantic")
        pixel_count = save_mask(predictions, args.class_id, out_file)
        total_pixels += pixel_count
        processed += 1

    elapsed = time.time() - start_all
    logger.info(
        "Saved %d masks to %s, skipped %d, total class pixels %d, elapsed %.2fs",
        processed,
        args.output,
        skipped,
        total_pixels,
        elapsed,
    )


if __name__ == "__main__":
    main()
