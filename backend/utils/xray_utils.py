"""Chest X-ray image preprocessing utilities."""

import io
import torch
from PIL import Image
from torchvision import transforms

IMG_SIZE = 128

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_image(raw_bytes: bytes) -> torch.Tensor:
    """
    raw_bytes → (1, 1, 128, 128) float32 tensor ready for XRayCNN.
    Accepts JPEG or PNG chest X-ray images.
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    tensor = INFERENCE_TRANSFORM(img)   # (1, 128, 128)
    return tensor.unsqueeze(0)          # (1, 1, 128, 128)
