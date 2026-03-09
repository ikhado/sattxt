# reference: https://github.com/IBM/MS-CLIP/blob/main/msclip/inference/utils.py

import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Union
import torch
import torch.nn.functional as F

def image_loader(path: Union[str, Path]) -> Image.Image:
    # reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


pretrained_data_stats = {
    'ms': {
        'means': [925.161, 1183.128, 1338.041, 1667.254, 2233.633, 2460.96, 2555.569, 2619.542, 2406.497, 1841.645],
        'stds': [1205.586, 1223.713, 1399.638, 1403.298, 1378.513, 1434.924, 1491.141, 1454.089, 1473.248, 1365.08],
        'size': [224]
    },
    'rgb': {
        'means': [0.48145466, 0.4578275, 0.40821073],
        'stds': [0.26862954, 0.26130258, 0.27577711],
        'size': [224]
    },
    'ms_all': {
        'means': [794.311, 925.161, 1183.128, 1338.041, 1667.254, 2233.633, 2460.96, 2555.569, 2619.542, 2703.298,
                  2406.497, 1841.645],
        'stds': [1164.883, 1205.586, 1223.713, 1399.638, 1403.298, 1378.513, 1434.924, 1491.141, 1454.089, 1660.395,
                 1473.248, 1365.08],
        'size': [224]
    }
}


def get_preprocess(is_ms=False, all_bands=False):
    # reference: https://github.com/IBM/MS-CLIP/blob/main/msclip/inference/utils.py
    if is_ms:
        if all_bands:
            data_params = pretrained_data_stats["ms_all"]
        else:
            data_params = pretrained_data_stats["ms"]
    else:
        data_params = pretrained_data_stats["rgb"]

    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: np.asarray(x, dtype=np.float32)),
        transforms.ToTensor(),  # for rgb the values are scaled but not for ms
        transforms.Resize(
            size=data_params["size"],
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(data_params["size"]),
        transforms.Normalize(mean=data_params["means"], std=data_params["stds"]),
    ])

    return preprocess


def zero_shot_classify(model, image: torch.Tensor, categories: list[str],
                       prompt_template: str = "a satellite image of {}"):
    """Zero-shot classification using text-image similarity."""
    with torch.no_grad():
        # Get vision features
        vision_features, _ = model({'image': image, 'caption': [prompt_template.format(categories[0])]})

        # Get text features for all categories
        text_features = []
        for cat in categories:
            _, feat = model({'image': image, 'caption': [prompt_template.format(cat)]})
            text_features.append(feat)
        text_features = torch.cat(text_features, dim=0)

        # Compute cosine similarity
        vision_norm = F.normalize(vision_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)
        logits = vision_norm @ text_norm.T

    return logits, logits.argmax(dim=-1)