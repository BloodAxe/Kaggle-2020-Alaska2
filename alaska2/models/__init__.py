import itertools
from typing import Tuple, Dict, Optional

import torch
from pytorch_toolbelt.inference.ensembling import Ensembler, ApplySigmoidTo, ApplySoftmaxTo
from torch import nn


from . import rgb_dct, rgb, dct, ela, rgb_ela_blur, timm, ycrcb, hpf_net, srnet
from ..dataset import *
from ..predict import *

MODEL_REGISTRY = {
    # TIMM
    "rgb_skresnext50_32x4d": timm.rgb_skresnext50_32x4d,
    "rgb_swsl_resnext101_32x8d": timm.rgb_swsl_resnext101_32x8d,
    "rgb_tf_efficientnet_b6_ns": timm.rgb_tf_efficientnet_b6_ns,
    "rgb_tf_efficientnet_b2_ns": timm.rgb_tf_efficientnet_b2_ns,
    "rgb_tresnet_m_448": timm.rgb_tresnet_m_448,
    "frank": rgb_ela_blur.frank,
    "rgb_dct_resnet34": rgb_dct.rgb_dct_resnet34,
    "rgb_dct_efficientb3": rgb_dct.rgb_dct_efficientb3,
    "rgb_dct_seresnext50": rgb_dct.rgb_dct_seresnext50,
    #
    "rgb_b0": rgb.rgb_b0,
    "rgb_resnet18": rgb.rgb_resnet18,
    "rgb_resnet34": rgb.rgb_resnet34,
    "rgb_seresnext50": rgb.rgb_seresnext50,
    "rgb_densenet121": rgb.rgb_densenet121,
    "rgb_densenet201": rgb.rgb_densenet201,
    "rgb_hrnet18": rgb.rgb_hrnet18,
    # DCT
    "dct_seresnext50": dct.dct_seresnext50,
    "dct_efficientnet_b6": dct.dct_efficientnet_b6,
    # ELA
    "ela_skresnext50_32x4d": ela.ela_skresnext50_32x4d,
    "ela_rich_skresnext50_32x4d": ela.ela_rich_skresnext50_32x4d,
    "ela_wider_resnet38": ela.ela_wider_resnet38,
    "ela_ecaresnext26tn_32x4d": ela.ela_ecaresnext26tn_32x4d,
    # YCrCb
    "ycrcb_skresnext50_32x4d": ycrcb.ycrcb_skresnext50_32x4d,
    "ela_s2d_skresnext50_32x4d": ycrcb.ela_s2d_skresnext50_32x4d,
    # HPF
    "hpf_net": hpf_net.hpf_net,
    "hpf_net2": hpf_net.hpf_net_v2,
    # SRNET
    "srnet": srnet.srnet,
    "srnet_inplace": srnet.srnet_inplace,
}

__all__ = ["MODEL_REGISTRY", "get_model", "ensemble_from_checkpoints", "wrap_model_with_tta"]


def get_model(model_name, num_classes=4, dropout=0, pretrained=True):
    return MODEL_REGISTRY[model_name](num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def model_from_checkpoint(model_checkpoint: str, model_name=None, report=True, strict=True) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    model_name = model_name or checkpoint["checkpoint_data"]["cmd_args"]["model"]

    model = get_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return model.eval(), checkpoint


def wrap_model_with_tta(model, tta_mode, inputs, outputs):
    if tta_mode == "flip-hv":
        model = HVFlipTTA(model, inputs=inputs, outputs=outputs, average=True)

    if tta_mode == "d4":
        model = D4TTA(model, inputs=inputs, outputs=outputs, average=True)

    return model


def ensemble_from_checkpoints(
    checkpoints, strict=True, outputs=None, activation: Optional[str] = "after_model", tta=None, temperature=1
):
    if activation not in {None, "after_model", "after_tta", "after_ensemble"}:
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
