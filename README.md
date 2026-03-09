# SATtxt - Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="SATtxt">
</p>

<p>
  <b> Minh Kha Do, Wei Xiang, Kang Han, Di Wu, Khoa Phan, Yi-Ping Phoebe Chen, Gaowen Liu, Ramana Rao Kompella </b>
</p>
<p>
  La Trobe University, Cisco Research
</p>

<p>
  <a href="https://arxiv.org/abs/2602.22613"><img src="https://img.shields.io/badge/arXiv-2602.22613-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/ikhado/sattxt"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
  <a href="https://ikhado.github.io/sattxt/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
</p>

---

## 📰 News

| Date | Update |
|------|--------|
| **Mar 9, 2026** | We have released model code and weights. |
| **Feb 23, 2026** | SATtxt is accepted at **CVPR 2026**. We appreciate the reviewers and ACs. |

---

## Overview

SATtxt is a vision-language foundation model for satellite imagery. We train **only the projection heads**, keeping both encoders frozen.

<table>
<tr><th>Component</th><th>Backbone</th><th>Parameters</th></tr>
<tr><td>Vision Encoder</td><td><a href="https://github.com/facebookresearch/dinov3">DINOv3</a> ViT-L/16</td><td>Frozen</td></tr>
<tr><td>Text Encoder</td><td><a href="https://github.com/McGill-NLP/llm2vec">LLM2Vec</a> Llama-3-8B</td><td>Frozen</td></tr>
<tr><td>Vision Head</td><td>Transformer Projection</td><td>Trained</td></tr>
<tr><td>Text Head</td><td>Linear Projection</td><td>Trained</td></tr>
</table>

---

## Installation

```bash
git clone https://github.com/ikhado/sattxt.git
cd sattxt
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # Required for LLM2Vec
```

---

## Model Weights

Download the required weights:

| Component | Source |
|-----------|--------|
| DINOv3 ViT-L/16 | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) → `dinov3_vitl16_pretrain_sat493m.pth` |
| LLM2Vec | [McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse) |
| Vision Head | [sattxt_vision_head.pt](https://huggingface.co/ikhado/sattxt/blob/main/sattxt_vision_head.pt) |
| Text Head | [sattxt_text_head.pt](https://huggingface.co/ikhado/sattxt/blob/main/sattxt_text_head.pt) |

Clone DINOv3 into the `thirdparty` folder:

```bash
cd thirdparty && git clone https://github.com/facebookresearch/dinov3.git
```

---

## Quick Start

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "thirdparty" / "dinov3"))

from sattxt.model import SATtxt
from sattxt.utils import image_loader, get_preprocess, zero_shot_classify
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = SATtxt(
    dinov3_weights_path="/PATH/TO/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    sattxt_vision_head_pretrain_weights="/PATH/TO/sattxt_vision_head.pt",
    text_encoder_id="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    sattxt_text_head_pretrain_weights="/PATH/TO/sattxt_text_head.pt",
).to(device).eval()

categories = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

image = image_loader("./asset/Residential_167.jpg")
image_tensor = get_preprocess(is_ms=False, all_bands=False)(image).unsqueeze(0).to(device)

logits, pred_idx = zero_shot_classify(model, image_tensor, categories)

print("Prediction:", categories[pred_idx.item()])
```

Please check [demo.py](./demo.py) for more details.

---

## Citation

```bibtex
@misc{do2026sattxt,
      title={Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery}, 
      author={Minh Kha Do and Wei Xiang and Kang Han and Di Wu and Khoa Phan and Yi-Ping Phoebe Chen and Gaowen Liu and Ramana Rao Kompella},
      year={2026},
      eprint={2602.22613},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.22613}, 
}
```

---

## Acknowledgements
We pretrained the model with:
[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)

We use evaluation scripts from:
[MS-CLIP](https://github.com/IBM/MS-CLIP) and [Pangaea-Bench](https://github.com/VMarsocci/pangaea-bench)

We also use LLMs (such as ChatGPT and Claude) for code refactoring.

---
<p>
  We welcome contributions and issues to further improve SATtxt.
</p>
