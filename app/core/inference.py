from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.core.config import IMG_SIZE, MEAN, STD

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    return TFM(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_probs(model, device, image: Image.Image) -> np.ndarray:
    x = preprocess(image).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probs