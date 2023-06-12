from typing import Tuple
from PIL import Image
import numpy as np


def letterbox_image(image: Image, size: Tuple[int, int]) -> Image:
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess(image: Image):
    img = np.array(letterbox_image(image, (500, 500))).astype(np.float32)
    # normalize to 0-1
    img /= 255.
    # normalize by mean + std
    img = (img - 0.48232) / np.array(0.2305)
    img = img.transpose((2, 0, 1))
    return img


def image_to_numpy(input_image: Image) -> np.ndarray:
    return preprocess(input_image)
