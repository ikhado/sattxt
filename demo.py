"""
SATtxt Demo: Zero-shot Land Cover Classification

This demo shows how to use SATtxt for zero-shot classification of satellite images.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "thirdparty" / "dinov3"))

from sattxt.model import SATtxt
from sattxt.utils import image_loader, get_preprocess, zero_shot_classify

def main():
    # Configuration - update these paths for your setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dinov3_weights_path = '/PATH/TO/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    sattxt_vision_head_weights = '/PATH/TO/sattxt_vision_head.pt'
    sattxt_text_head_weights = '/PATH/TO/sattxt_text_head.pt'
    text_encoder_id = 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp'

    # Load model
    model = SATtxt(
        dinov3_weights_path=dinov3_weights_path,
        sattxt_vision_head_pretrain_weights=sattxt_vision_head_weights,
        text_encoder_id=text_encoder_id,
        sattxt_text_head_pretrain_weights=sattxt_text_head_weights
    ).to(device).eval()

    # Land cover categories (EuroSAT)
    categories = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]

    # Load and preprocess image
    image_path = './asset/Residential_167.jpg'
    image = image_loader(image_path)
    preprocess = get_preprocess(is_ms=False, all_bands=False)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Zero-shot classification
    logits, pred_idx = zero_shot_classify(model, image_tensor, categories)
    
    # Results
    print(f"Image: {image_path}")
    print(f"Predicted: {categories[pred_idx.item()]}")
    print(f"Confidence scores:")
    for cat, score in zip(categories, logits.squeeze().tolist()):
        print(f"  {cat}: {score:.4f}")


if __name__ == '__main__':
    main()
