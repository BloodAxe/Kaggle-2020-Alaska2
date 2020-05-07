import torch
from pytorch_toolbelt.inference.ensembling import Ensembler, ApplySigmoidTo, ApplySoftmaxTo
from torch import nn


from . import rgb_dct, rgb, dct, ela, rgb_ela_blur
from ..dataset import *
from ..predict import *

MODEL_REGISTRY = {
    "rgb_dct_resnet34": rgb_dct.rgb_dct_resnet34,
    "rgb_dct_efficientb3": rgb_dct.rgb_dct_efficientb3,
    "rgb_dct_seresnext50": rgb_dct.rgb_dct_seresnext50,
    "rgb_dct_b0_srnet": rgb_dct.rgb_dct_b0_srnet,
    #
    "rgb_resnet18": rgb.rgb_resnet18,
    "rgb_resnet34": rgb.rgb_resnet34,
    "dct_resnet34": dct.dct_resnet34,
    "ela_resnet34": ela.ela_resnet34,
    # RGB + ELA + BLUR
    "rgb_ela_blur_resnet18": rgb_ela_blur.rgb_ela_blur_resnet18,
}

__all__ = ["MODEL_REGISTRY", "get_model", "ensemble_from_checkpoints", "wrap_model_with_tta"]


def get_model(model_name, num_classes=4, dropout=0, pretrained=True):
    return MODEL_REGISTRY[model_name](num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def model_from_checkpoint(model_checkpoint: str, model_name=None, report=True, strict=True) -> nn.Module:
    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    model_name = model_name or checkpoint["checkpoint_data"]["cmd_args"]["model"]

    model = get_model(model_name, pretrained=strict)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return model.eval(), checkpoint


def wrap_model_with_tta(model, tta_mode, outputs):
    if tta_mode == "d4":
        model = D4TTA(model, outputs, average=False)

    if tta_mode == "d4-avg":
        model = D4TTA(model, outputs, average=True)

    if tta_mode == "ms":
        model = MultiscaleTTA(model, outputs, size_offsets=[-64, 64], average=False)

    if tta_mode == "rot-avg":
        model = Rot180TTA(model, outputs, average=True)

    if tta_mode == "ms-avg":
        model = MultiscaleTTA(model, outputs, size_offsets=[-64, 64], average=True)

    if tta_mode == "zoom-in-avg":
        model = MultiscaleTTA(model, outputs, size_offsets=[-64, -128], average=True)

    if tta_mode == "zoom-out-avg":
        model = MultiscaleTTA(model, outputs, size_offsets=[+64, +128], average=True)

    if tta_mode == "rot-flip-avg":
        model = HFlipTTA(model, outputs, average=True)
        model = Rot180TTA(model, outputs, average=True)

    if tta_mode == "ms-rot-avg":
        model = Rot180TTA(model, outputs, average=True)
        model = MultiscaleTTA(model, outputs, size_offsets=[-128, -64, 64, +128], average=True)

    if tta_mode == "ms-rot-flip-avg":
        model = HFlipTTA(model, outputs, average=True)
        model = Rot180TTA(model, outputs, average=True)
        model = MultiscaleTTA(model, outputs, size_offsets=[-128, -64, 64, +128], average=True)

    if tta_mode == "ms2-rot-flip-avg":
        model = HFlipTTA(model, outputs, average=True)
        model = Rot180TTA(model, outputs, average=True)
        model = MultiscaleTTA(model, outputs, size_offsets=[-128, -64, 64, +128, +192], average=True)

    if tta_mode == "ms-rot-flip":
        model = HFlipTTA(model, outputs, average=False)
        model = Rot180TTA(model, outputs, average=False)
        model = MultiscaleTTA(model, outputs, size_offsets=[-128, -64, 64, +128], average=True)

    if tta_mode == "ms-d4":
        model = D4TTA(model, outputs, average=False)
        model = MultiscaleTTA(model, outputs, size_offsets=[-64, 64], average=False)

    if tta_mode == "ms-d4-avg":
        model = D4TTA(model, outputs, average=True)
        model = MultiscaleTTA(model, outputs, size_offsets=[-64, 64], average=True)

    return model


def ensemble_from_checkpoints(checkpoints, strict=False, outputs=None, activation: str = "after_model", tta=None):
    if activation not in {"after_model", "after_tta", "after_ensemble"}:
        raise KeyError(activation)

    models, loaded_checkpoints = zip(*[model_from_checkpoint(ck, strict=strict) for ck in checkpoints])

    if activation == "after_model":
        models = [ApplySigmoidTo(m, output_key=OUTPUT_PRED_MODIFICATION_FLAG) for m in models]
        models = [ApplySoftmaxTo(m, output_key=OUTPUT_PRED_MODIFICATION_TYPE) for m in models]
        print("Applying sigmoid activation to outputs", outputs, "after model")

    if len(models) > 1:
        model = Ensembler(models, outputs=outputs)
        if activation == "after_ensemble":
            model = ApplySigmoidTo(model, output_key=OUTPUT_PRED_MODIFICATION_FLAG)
            model = ApplySoftmaxTo(model, output_key=OUTPUT_PRED_MODIFICATION_TYPE)
            print("Applying sigmoid activation to outputs", outputs, "after ensemble")
    else:
        assert len(models) == 1
        model = models[0]

    if tta is not None:
        model = wrap_model_with_tta(model, tta, outputs=outputs)
        print("Wrapping models with TTA", tta)

    if activation == "after_tta":
        models = [ApplySigmoidTo(m, output_key=OUTPUT_PRED_MODIFICATION_FLAG) for m in models]
        models = [ApplySoftmaxTo(m, output_key=OUTPUT_PRED_MODIFICATION_TYPE) for m in models]
        print("Applying sigmoid activation to outputs", outputs, "after TTA")

    return model.eval(), loaded_checkpoints
