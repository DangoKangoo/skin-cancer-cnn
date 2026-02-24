from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from app.core.config import IMG_SIZE
from app.core.inference import preprocess


def compute_gradcam_mobilenetv2(model, device, image: Image.Image, target_class: int) -> np.ndarray:
    target_layer = model.features[-1]
    activations = None
    gradients = None

    def fwd_hook(_m, _i, o):
        nonlocal activations
        activations = o

    def bwd_hook(_m, _gin, gout):
        nonlocal gradients
        gradients = gout[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        x = preprocess(image).to(device)
        x.requires_grad_(True)

        logits = model(x)
        score = logits[0, target_class]

        model.zero_grad(set_to_none=True)
        score.backward()

        w = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.astype(np.float32)
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    base_np = np.array(base).astype(np.float32)

    heat = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
    heat_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    heat_rgb[..., 0] = heat

    out = (1 - alpha) * base_np + alpha * heat_rgb.astype(np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)