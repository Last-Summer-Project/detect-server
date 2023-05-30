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


def image_to_numpy(input_image: Image) -> np.ndarray:
    return (np.array(letterbox_image(input_image, (500, 500)))  # letter boxing
            .astype(np.float32)  # convert to float
            .transpose((-1, 0, 1)))  # transpose to correct input
