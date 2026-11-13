"""I-JEPA image encoder from Hugging Face transformers.

An image becomes one vector: the mean of I-JEPA's last hidden state over all
patch tokens (1280-d for ViT-H/14).
"""

from __future__ import annotations

import requests
import torch
from PIL import Image

MODEL_ID = "facebook/ijepa_vith14_1k"

# Some image hosts reject requests without a browser-like User-Agent. The first
# notebook attempt (Wikimedia URLs, no header) failed to download every image.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


class IJepaEncoder:
    """Loads I-JEPA once and turns images into mean-pooled embeddings."""

    def __init__(self, model_id: str = MODEL_ID):
        from transformers import AutoModel, AutoProcessor  # heavy import, keep it lazy

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(image.convert("RGB"), return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def encode_url(self, url: str, timeout: float = 15) -> torch.Tensor:
        response = requests.get(url, headers=_HEADERS, stream=True, timeout=timeout)
        response.raise_for_status()
        return self.encode_image(Image.open(response.raw))
