"""Check webcam-independent web inference without changing the author model."""

import json
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
MODEL_DIR = WORKSPACE / "Earth2Ocean-code"
OUT = Path(__file__).resolve().parent
os.chdir(MODEL_DIR)
sys.path.insert(0, str(MODEL_DIR))

import mmcv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from environment.windows_compat import prepare_mmseg

prepare_mmseg()
from Earth2Ocean import Earth2OceanSegmentation, openai_underwater_template
from mmseg.structures import SegDataSample

CLASSES = ["underwater background", "clownfish", "octopus", "sea turtle"]
SOURCES = [("ClownFish", 3), ("Octopus", 20), ("Turtle", 36)]
PALETTE = [(23, 28, 38), (255, 129, 72), (211, 137, 247), (72, 227, 174)]
FONT = ImageFont.truetype("arial.ttf", 19)


def select_case(prefix, expected_id):
    root = MODEL_DIR / "datasets" / "MAS3K"
    for path in sorted((root / "img").glob(prefix + "_*.jpg"))[:40]:
        mask = root / "mask" / (path.stem + ".png")
        if mask.is_file():
            target = np.asarray(Image.open(mask)) == expected_id
            if 0.10 <= target.mean() <= 0.65:
                return path, target
    raise RuntimeError(f"No eligible sample in {prefix}")


@torch.inference_mode()
def set_query_words(model, words, cache):
    # Encode the same underwater templates as the constructor; retain the loaded models.
    for word in words:
        if word not in cache:
            tokens = model.tokenizer([tpl(word) for tpl in openai_underwater_template])
            features = model.clip.encode_text(tokens.to(model.device))
            average = torch.nn.functional.normalize(features, dim=-1).mean(dim=0)
            cache[word] = torch.nn.functional.normalize(average, dim=0)
    model.query_features = torch.stack([cache[word] for word in words])
    model.query_idx = torch.arange(len(words), device=model.device)
    model.num_queries = model.num_classes = len(words)


@torch.inference_mode()
def infer(model, rgb, upload_filename):
    original = np.array(rgb)
    h, w = original.shape[:2]
    frame = mmcv.imrescale(original[:, :, ::-1].copy(), (2048, 336))
    sample = SegDataSample(metainfo={
        "ori_shape": (h, w),
        "img_shape": frame.shape[:2],
        "img_path": upload_filename,
    })
    batch = dict(inputs=[torch.from_numpy(frame).permute(2, 0, 1)], data_samples=[sample])
    prepared = model.data_preprocessor(batch, training=False)
    prediction = model.predict(prepared["inputs"], prepared["data_samples"])[0]
    ids = prediction.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
    assert ids.shape == (h, w)
    return ids


