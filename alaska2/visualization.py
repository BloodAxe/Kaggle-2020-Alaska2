import cv2
import numpy as np
import torch
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor, to_numpy
from pytorch_toolbelt.utils.catalyst import draw_binary_segmentation_predictions

from .dataset import *

__all__ = ["draw_predictions"]


@torch.no_grad()
def draw_predictions(input: dict, output: dict, mean=0.0, std=1.0, max_images=16):
    images = []

    num_images = len(input[INPUT_IMAGE_ID_KEY])
    for i in range(num_images):
        image_id = input[INPUT_IMAGE_ID_KEY][i]
        if INPUT_IMAGE_KEY in input:
            image = rgb_image_from_tensor(input[INPUT_IMAGE_KEY][i], mean, std, max_pixel_value=1)
        elif INPUT_FEATURES_JPEG_FLOAT in input:
            image = rgb_image_from_tensor(input[INPUT_FEATURES_JPEG_FLOAT][i], mean, std, max_pixel_value=1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = image.copy()
        true_type = int(input[INPUT_TRUE_MODIFICATION_TYPE][i])
        true_flag = float(input[INPUT_TRUE_MODIFICATION_FLAG][i])
        pred_type = int(output[OUTPUT_PRED_MODIFICATION_TYPE][i].argmax())
        pred_flag = float(output[OUTPUT_PRED_MODIFICATION_FLAG][i].sigmoid())

        header = np.zeros((40, overlay.shape[1], 3), dtype=np.uint8) + 40
        cv2.putText(header, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))
        cv2.putText(
            header,
            f"true_type={true_type}/pred_type={pred_type} true_flag={true_flag:.2f}/pred_flag={pred_flag:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (250, 250, 250),
        )

        overlay = np.row_stack([header, overlay])

        if INPUT_TRUE_MODIFICATION_MASK in input and OUTPUT_PRED_MODIFICATION_MASK in output:
            true_mask = to_numpy(input[INPUT_TRUE_MODIFICATION_MASK][i, 0] > 0)
            pred_mask = to_numpy(output[OUTPUT_PRED_MODIFICATION_MASK][i, 0] > 0)

            mask_overlay = image.copy()

            mask_overlay[true_mask & pred_mask] = np.array(
                [0, 250, 0], dtype=overlay.dtype
            )  # Correct predictions (Hits) painted with green
            mask_overlay[true_mask & ~pred_mask] = np.array(
                [250, 0, 0], dtype=overlay.dtype
            )  # Misses painted with red
            mask_overlay[~true_mask & pred_mask] = np.array(
                [250, 250, 0], dtype=overlay.dtype
            )  # False alarm painted with yellow
            mask_overlay = cv2.addWeighted(image, 0.5, mask_overlay, 0.5, 0, dtype=cv2.CV_8U)
            mask_overlay = np.row_stack([mask_overlay, overlay])

            overlay = np.column_stack([overlay, mask_overlay])

        images.append(overlay)
        if len(images) >= max_images:
            break

    return images
