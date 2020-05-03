import cv2
import numpy as np
import torch
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor

from .dataset import *

__all__ = ["draw_predictions"]


@torch.no_grad()
def draw_predictions(input: dict, output: dict, mean=0.0, std=1.0, max_images=16):
    images = []

    num_images = len(input[INPUT_IMAGE_ID_KEY])
    for i in range(num_images):
        image_id = input[INPUT_IMAGE_ID_KEY][i]
        image = rgb_image_from_tensor(input[INPUT_IMAGE_KEY][i], mean, std, max_pixel_value=1)
        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        true_label = int(input[INPUT_TRUE_MODIFICATION_TYPE][i])
        pred_type = int(output[OUTPUT_PRED_MODIFICATION_TYPE][i].argmax())
        pred_flag = float(output[OUTPUT_PRED_MODIFICATION_FLAG][i].sigmoid())

        header = np.zeros((40, overlay.shape[1], 3), dtype=np.uint8) + 40
        cv2.putText(header, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))
        cv2.putText(
            header, f"{true_label}/{pred_type}/{pred_flag:.4f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250)
        )

        overlay = np.row_stack([header, overlay])
        images.append(overlay)
        if len(images) >= max_images:
            break

    return images
