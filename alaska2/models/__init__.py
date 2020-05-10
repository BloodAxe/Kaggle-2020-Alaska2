import itertools
from typing import Tuple, Dict

import torch
from pytorch_toolbelt.inference.ensembling import Ensembler, ApplySigmoidTo, ApplySoftmaxTo
from torch import nn


from . import rgb_dct, rgb, dct, ela, rgb_ela_blur, timm
from ..dataset import *
from ..predict import *

MODEL_REGISTRY = {
    # TIMM
    "rgb_skresnext50_32x4d": timm.rgb_skresnext50_32x4d,
    "rgb_dpn92": timm.rgb_dpn92,

    "frank": rgb_ela_blur.frank,
    "rgb_dct_resnet34": rgb_dct.rgb_dct_resnet34,
    "rgb_dct_efficientb3": rgb_dct.rgb_dct_efficientb3,
    "rgb_dct_seresnext50": rgb_dct.rgb_dct_seresnext50,
    "rgb_dct_b0_srnet": rgb_dct.rgb_dct_b0_srnet,
    #
    "rgb_b0": rgb.rgb_b0,
    "rgb_resnet18": rgb.rgb_resnet18,
    "rgb_resnet34": rgb.rgb_resnet34,
    "rgb_seresnext50": rgb.rgb_seresnext50,
    "rgb_densenet121": rgb.rgb_densenet121,
    "rgb_densenet201": rgb.rgb_densenet201,
    "rgb_hrnet18": rgb.rgb_hrnet18,

    # DCT
    "dct_resnet34": dct.dct_resnet34,
    "ela_resnet34": ela.ela_resnet34,

}

__all__ = ["MODEL_REGISTRY", "get_model", "ensemble_from_checkpoints", "wrap_model_with_tta"]


def get_model(model_name, num_classes=4, dropout=0, pretrained=True):
    return MODEL_REGISTRY[model_name](num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def model_from_checkpoint(model_checkpoint: str, model_name=None, report=True, strict=True) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    model_name = model_name or checkpoint["checkpoint_data"]["cmd_args"]["model"]

    model = get_model(model_name, pretrained=strict)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return model.eval(), checkpoint


def wrap_model_with_tta(model, tta_mode, inputs, outputs):
    if tta_mode == "flip-hv":
        model = HVFlipTTA(model, inputs=inputs, outputs=outputs, average=True)

    return model


def ensemble_from_checkpoints(
    checkpoints, strict=False, outputs=None, activation: str = "after_model", tta=None, temperature=1
):
    if activation not in {"after_model", "after_tta", "after_ensemble"}:
        raise KeyError(activation)

    models, loaded_checkpoints = zip(*[model_from_checkpoint(ck, strict=strict) for ck in checkpoints])

    required_features = itertools.chain(*[m.required_features for m in models])
    required_features = list(set(list(required_features)))

    if activation == "after_model":
        models = [ApplySigmoidTo(m, output_key=OUTPUT_PRED_MODIFICATION_FLAG, temperature=temperature) for m in models]
        models = [ApplySoftmaxTo(m, output_key=OUTPUT_PRED_MODIFICATION_TYPE, temperature=temperature) for m in models]
        print("Applying sigmoid activation to outputs", outputs, "after model")

    if len(models) > 1:

        model = Ensembler(models, outputs=outputs)
        if activation == "after_ensemble":
            model = ApplySigmoidTo(model, output_key=OUTPUT_PRED_MODIFICATION_FLAG, temperature=temperature)
            model = ApplySoftmaxTo(model, output_key=OUTPUT_PRED_MODIFICATION_TYPE, temperature=temperature)
            print("Applying sigmoid activation to outputs", outputs, "after ensemble")
    else:
        assert len(models) == 1
        model = models[0]

    if tta is not None:
        model = wrap_model_with_tta(model, tta, inputs=required_features, outputs=outputs)
        print("Wrapping models with TTA", tta)

    if activation == "after_tta":
        model = ApplySigmoidTo(model, output_key=OUTPUT_PRED_MODIFICATION_FLAG, temperature=temperature)
        model = ApplySoftmaxTo(model, output_key=OUTPUT_PRED_MODIFICATION_TYPE, temperature=temperature)
        print("Applying sigmoid activation to outputs", outputs, "after TTA")

    return model.eval(), loaded_checkpoints, required_features
