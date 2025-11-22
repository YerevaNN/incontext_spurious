"""Implements SigLIP2 encoding extractor."""

from PIL import Image

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image


class SigLIP2(torch.nn.Module):
    def __init__(self, ckpt, *args, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(ckpt).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)

    def forward(self, images):
        def to_image(x):
            x = x.cpu().numpy()
            x = x.transpose((1, 2, 0))
            x = (x * 255).astype(np.uint8)
            return Image.fromarray(x)

        images = [to_image(x) for x in images]
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)

        return image_embeddings  # [batch_size, hidden_size]

