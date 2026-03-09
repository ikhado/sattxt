# this file is adapted from https://github.com/IBM/MS-CLIP/blob/main/msclip/inference/utils.py
import numpy as np
from torchvision import transforms

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


def get_preprocess(is_ms: bool = False, all_bands: bool = False):
    stats_key = "rgb"
    if is_ms:
        stats_key = "ms_all" if all_bands else "ms"
    data_params = pretrained_data_stats[stats_key]

    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.astype(np.float32)),
        transforms.ToTensor(),  # for rgb the values are scaled but not for ms
        transforms.Resize(
            size=data_params["size"],
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(data_params["size"]),
        transforms.Normalize(mean=data_params["means"], std=data_params["stds"]),
    ])

    return preprocess
