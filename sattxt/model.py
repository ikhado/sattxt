from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from sattxt.modules import LinearProj
from dinov3.eval.text.vision_tower import build_vision_model
from llm2vec import LLM2Vec

DINO_REPO_DIR = str(Path(__file__).resolve().parent.parent / "thirdparty" / "dinov3")


class SATtxt(nn.Module):
    def __init__(
        self,
        dinov3_weights_path: Optional[str] = None,
        sattxt_vision_head_pretrain_weights: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        sattxt_text_head_pretrain_weights: Optional[str] = None,
    ):
        super().__init__()

        self.embed_dim = 2048
        self.visual = VisionEncoder(
            dinov3_weights_path=dinov3_weights_path,
            output_dim=self.embed_dim,
            sattxt_vision_head_pretrain_weights=sattxt_vision_head_pretrain_weights,
        )
        self.text = TextEncoder(
            text_encoder_id=text_encoder_id,
            output_dim=self.embed_dim,
            sattxt_text_head_pretrain_weights=sattxt_text_head_pretrain_weights,
        )

    def forward(self, x):
        images, captions = x['image'], x['caption']  # image: (B, Cin, H, W) with Cin = 3

        encoded_img = self.visual(images)
        encoded_text = self.text(captions)

        return encoded_img, encoded_text


class VisionEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dinov3_weights_path: Optional[str],
        sattxt_vision_head_pretrain_weights: Optional[str],
    ):
        super().__init__()
        self.encoder = None
        self.embed_dim = output_dim
        self.setup_image_encoder(dinov3_weights_path, sattxt_vision_head_pretrain_weights)

    def setup_image_encoder(
        self,
        dinov3_weights_path: Optional[str] = None,
        sattxt_vision_head_pretrain_weights: Optional[str] = None,
    ):
        backbone = torch.hub.load(DINO_REPO_DIR, 'dinov3_vitl16', source='local', weights=dinov3_weights_path)

        self.encoder = build_vision_model(
            embed_dim=self.embed_dim,
            backbone_model_config=None,
            freeze_backbone=True,
            num_head_blocks=2,
            blocks_drop_path=0.3,
            use_class_token=True,
            use_patch_tokens=True,
            patch_token_layer=1,
            patch_tokens_pooler_type="mean",
            use_linear_projection=False,
            backbone=backbone,
        )

        if sattxt_vision_head_pretrain_weights is not None:
            head_weights = torch.load(sattxt_vision_head_pretrain_weights, map_location="cpu", weights_only=False)
            msg = self.encoder.head.load_state_dict(head_weights, strict=True)
            print('loaded vision head weights: {msg}'.format(msg=msg))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # DINOv3txt returns a tuple where index 0 is the concatenated global embedding.
        return self.encoder(image)[0]


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_id: Optional[str],
        output_dim: int,
        sattxt_text_head_pretrain_weights: Optional[str],
    ):
        super().__init__()
        self.text_hidden_dim = 4096
        self.output_dim = output_dim
        self.text_encoder_id = text_encoder_id
        self.encoder: Optional[LLM2Vec] = None
        self.setup_text_encoder()
        self.text_projection = LinearProj(in_dim=self.text_hidden_dim, out_dim=self.output_dim)
        if sattxt_text_head_pretrain_weights is not None:
            head_weights = torch.load(sattxt_text_head_pretrain_weights, map_location="cpu", weights_only=False)
            msg = self.text_projection.load_state_dict(head_weights, strict=True)
            print('loaded text head weights: {msg}'.format(msg=msg))

    def setup_text_encoder(self):
        if self.encoder is None:
            if not self.text_encoder_id:
                raise ValueError("text_encoder_id must be provided to initialize LLM2Vec")

            encoder = LLM2Vec.from_pretrained(
                self.text_encoder_id,
                peft_model_name_or_path=self.text_encoder_id + "-unsup-simcse",
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                pooling_mode="mean",
                max_length=512
            )

            self.encoder = encoder

    def forward(self, captions: List[str]) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Text encoder is not initialized")

        device = next(self.parameters()).device
        vec = self.encoder.encode(captions, show_progress_bar=False,
                                  convert_to_tensor=True, device=device)  # (B, 4096) NumPy or Tensor
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec)
        vec = vec.to(device)
        vec = self.text_projection(vec)

        return vec
