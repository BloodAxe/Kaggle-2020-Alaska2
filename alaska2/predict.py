from typing import List

import pytorch_toolbelt.inference.functional as AF
import torch
import torch.nn.functional as F
from torch import nn

from .dataset import *

__all__ = ["HVFlipTTA", "D4TTA"]


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
        self.output_keys = outputs
        self.average = average

    def augment_inputs(self, augment_fn, kwargs):
        augmented_inputs = dict(
            (key, augment_fn(value) if key in self.inputs else value) for key, value in kwargs.items()
        )
        return augmented_inputs

    def forward(self, **kwargs):

        normal_input = kwargs
        outputs = self.model(**normal_input)

        other_outputs = [
            self.model(**self.augment_inputs(AF.torch_fliplr, normal_input)),
            self.model(**self.augment_inputs(AF.torch_flipud, normal_input)),
            self.model(**self.augment_inputs(torch_flip_ud_lr, normal_input)),
        ]

        # Create extra output with _tta suffix that contains contatenated predictions
        for output_key in self.output_keys:
            tta_outputs = [outputs[output_key]] + [out[output_key] for out in other_outputs]
            outputs[output_key + "_tta"] = torch.cat(tta_outputs, dim=1)

        for tta_outputs in other_outputs:
            for output_key in self.output_keys:
                outputs[output_key] += tta_outputs[output_key]

        scale = 1.0 / (1 + len(other_outputs))
        for output_key in self.output_keys:
            outputs[output_key] *= scale

        return outputs


class D4TTA(nn.Module):
    def __init__(self, model, inputs, outputs, average=True):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.output_keys = outputs
        self.average = average

    def augment_inputs(self, augment_fn, kwargs):
        augmented_inputs = dict(
            (key, augment_fn(value) if key in self.inputs else value) for key, value in kwargs.items()
        )
        return augmented_inputs

    def forward(self, **kwargs):
        normal_input = kwargs
        fliped_input = self.augment_inputs(AF.torch_transpose, kwargs)

        outputs = self.model(**normal_input)

        other_outputs = [
            self.model(**self.augment_inputs(AF.torch_rot90, normal_input)),
            self.model(**self.augment_inputs(AF.torch_rot180, normal_input)),
            self.model(**self.augment_inputs(AF.torch_rot270, normal_input)),
            self.model(**fliped_input),
            self.model(**self.augment_inputs(AF.torch_rot90, fliped_input)),
            self.model(**self.augment_inputs(AF.torch_rot180, fliped_input)),
            self.model(**self.augment_inputs(AF.torch_rot270, fliped_input)),
        ]

        # Create extra output with _tta suffix that contains contatenated predictions
        for output_key in self.output_keys:
            tta_outputs = [outputs[output_key]] + [out[output_key] for out in other_outputs]
            outputs[output_key + "_tta"] = torch.cat(tta_outputs, dim=1)

        for tta_outputs in other_outputs:
            for output_key in self.output_keys:
                outputs[output_key] += tta_outputs[output_key]

        scale = 1.0 / (1 + len(other_outputs))
        for output_key in self.output_keys:
            outputs[output_key] *= scale

        return outputs
