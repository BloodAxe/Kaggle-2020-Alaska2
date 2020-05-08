from typing import List

import pytorch_toolbelt.inference.functional as AF
import torch
import torch.nn.functional as F
from torch import nn

from .dataset import *

__all__ = [
    "HVFlipTTA",
    "Rot180TTA",
    "D4TTA",
    "predict_from_flag",
    "predict_from_flag_and_type_sum",
    "predict_from_type",
]

def torch_flip_ud_lr(x: torch.Tensor):
    """
    Flip 4D image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2).flip(3)

class HVFlipTTA(nn.Module):
    def __init__(self, model, inputs, outputs, average=True):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.average = average

    def augment_inputs(self, augment_fn, kwargs):
        augmented_inputs = dict(
            (key, augment_fn(value) if key in self.inputs else value) for key, value in kwargs.items()
        )
        return augmented_inputs

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        outputs_lr = self.model(**self.augment_inputs(AF.torch_fliplr, kwargs))
        outputs_ud = self.model(**self.augment_inputs(AF.torch_flipud, kwargs))
        outputs_hv = self.model(**self.augment_inputs(torch_flip_ud_lr, kwargs))

        for output_key in self.outputs:
            outputs[output_key] += outputs_lr[output_key]
            outputs[output_key] += outputs_ud[output_key]
            outputs[output_key] += outputs_hv[output_key]

        scale = 0.25
        for output_key in self.outputs:
            outputs[output_key] *= scale

        return outputs


class Rot180TTA(nn.Module):
    def __init__(self, model, outputs, average=True):
        super().__init__()
        self.model = model
        self.outputs = outputs
        self.average = average

    def forward(self, image):
        outputs = self.model(image)

        augment = [AF.torch_rot180]
        deaugment = [AF.torch_rot180]

        for aug, deaug in zip(augment, deaugment):
            input = aug(image)
            aug_output = self.model(input)

            for output_key in self.outputs:
                outputs[output_key] += deaug(aug_output[output_key])

        if self.average:
            averaging_scale = 1.0 / 2.0
            for output_key in self.outputs:
                outputs[output_key] *= averaging_scale

        return outputs


class D4TTA(nn.Module):
    def __init__(self, model, outputs, average=True):
        super().__init__()
        self.model = model
        self.outputs = outputs
        self.average = average

    def forward(self, image):
        outputs = self.model(image)

        augment = [AF.torch_rot90, AF.torch_rot180, AF.torch_rot270]
        deaugment = [AF.torch_rot270, AF.torch_rot180, AF.torch_rot90]

        for aug, deaug in zip(augment, deaugment):
            input = aug(image)
            aug_output = self.model(input)

            for output_key in self.outputs:
                outputs[output_key] += deaug(aug_output[output_key])

        image_t = AF.torch_transpose(image)

        augment = [AF.torch_none, AF.torch_rot90, AF.torch_rot180, AF.torch_rot270]
        deaugment = [AF.torch_none, AF.torch_rot270, AF.torch_rot180, AF.torch_rot90]

        for aug, deaug in zip(augment, deaugment):
            input = aug(image_t)
            aug_output = self.model(input)

            for output_key in self.outputs:
                x = deaug(aug_output[output_key])
                outputs[output_key] += AF.torch_transpose(x)

        if self.average:
            averaging_scale = 1.0 / 8.0
            for output_key in self.outputs:
                outputs[output_key] *= averaging_scale

        return outputs


def predict_from_flag(model, inputs):
    outputs = model(**inputs)
    probs = outputs[OUTPUT_PRED_MODIFICATION_FLAG]
    return probs


def predict_from_flag_and_type_sum(model, inputs):
    outputs = model(**inputs)
    flag_prob = outputs[OUTPUT_PRED_MODIFICATION_FLAG]
    no_mod_type = outputs[OUTPUT_PRED_MODIFICATION_TYPE][:, 0:1]
    has_mod_type = outputs[OUTPUT_PRED_MODIFICATION_TYPE][:, 1:].sum(dim=1, keepdim=True)
    return torch.clamp(flag_prob - no_mod_type + has_mod_type, 0, 1)


def predict_from_type(model, inputs):
    outputs = model(**inputs)
    no_mod_type = outputs[OUTPUT_PRED_MODIFICATION_TYPE][:, 0:1]
    has_mod_type = outputs[OUTPUT_PRED_MODIFICATION_TYPE][:, 1:].sum(dim=1, keepdim=True)
    return has_mod_type
