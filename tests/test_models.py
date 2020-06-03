import pytest, os, cv2, torch
from pytorch_toolbelt.utils import tensor_from_rgb_image, count_parameters
from pytorch_toolbelt.optimization.lr_schedules import CosineAnnealingLRWithDecay, OnceCycleLR, PolyLR
from torch import nn
import matplotlib.pyplot as plt
from torch.optim import SGD, Optimizer

from alaska2 import *
from alaska2.models.dct import dct_seresnext50
from alaska2.models.hpf_net import HPFNet
from alaska2.models.srnet import SRNetModel, srnet
from alaska2.models.ycrcb import ela_s2d_skresnext50_32x4d

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

KNOWN_KEYS = ["encoder", "rgb_encoder", "dct_encoder", "embedding", "type_classifier", "flag_classifier"]


@torch.no_grad()
def test_hpfnet():
    model = HPFNet(num_classes=4).cuda()

    input = {INPUT_IMAGE_KEY: torch.randn((2, 3, 512, 512)).cuda()}

    print(count_parameters(model, keys=KNOWN_KEYS))

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())


@torch.no_grad()
def test_srnet():
    model = srnet().cuda()

    input = {INPUT_IMAGE_KEY: torch.randn((2, 3, 512, 512)).cuda()}

    print(count_parameters(model, keys=KNOWN_KEYS))

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())


@torch.no_grad()
def test_ela_s2d_skresnext50_32x4d():
    model = ela_s2d_skresnext50_32x4d().cuda()

    input = {
        INPUT_FEATURES_CHANNEL_Y_KEY: torch.randn((2, 1, 512, 512)).cuda(),
        INPUT_FEATURES_CHANNEL_CB_KEY: torch.randn((2, 1, 512, 512)).cuda(),
        INPUT_FEATURES_CHANNEL_CR_KEY: torch.randn((2, 1, 512, 512)).cuda(),
        INPUT_FEATURES_ELA_RICH_KEY: torch.randn((2, 9, 512, 512)).cuda(),
    }

    print(count_parameters(model, keys=KNOWN_KEYS))

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())


@torch.no_grad()
def test_ela_wider_resnet38():
    model = get_model("ela_wider_resnet38").cuda().eval()
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    ela = compute_ela(image)
    blur = compute_blur_features(image)

    print(count_parameters(model, keys=KNOWN_KEYS))
    input = {
        INPUT_IMAGE_KEY: tensor_from_rgb_image(image).unsqueeze(0).cuda(),
        INPUT_FEATURES_BLUR_KEY: tensor_from_rgb_image(blur).unsqueeze(0).cuda().float(),
        INPUT_FEATURES_ELA_KEY: tensor_from_rgb_image(ela).unsqueeze(0).cuda().float(),
    }

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())


@torch.no_grad()
def test_dct_seresnext50():
    model = dct_seresnext50().cuda()

    input = {
        INPUT_FEATURES_DCT_KEY: torch.randn((2, 64 * 3, 512 // 8, 512 // 8)).cuda(),
    }

    print(count_parameters(model, keys=KNOWN_KEYS))

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())


@torch.no_grad()
@pytest.mark.parametrize("model_name", MODEL_REGISTRY.keys())
def test_models_forward(model_name):
    model = get_model(model_name).cuda().eval()
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    dct = compute_dct(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    ela = compute_ela(image)
    blur = compute_blur_features(image)

    print(count_parameters(model, keys=KNOWN_KEYS))
    input = {
        INPUT_IMAGE_KEY: tensor_from_rgb_image(image).unsqueeze(0).cuda(),
        INPUT_FEATURES_DCT_KEY: tensor_from_rgb_image(dct).unsqueeze(0).cuda().float(),
        INPUT_FEATURES_BLUR_KEY: tensor_from_rgb_image(blur).unsqueeze(0).cuda().float(),
        INPUT_FEATURES_ELA_KEY: tensor_from_rgb_image(ela).unsqueeze(0).cuda().float(),
    }

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())


def test_plot_lr():

    plt.figure(figsize=(16, 20))

    epochs = 100
    schedulers = [
        # "cosd",
        # "cosr",
        # "cosrd",
        "flat_cos",
        "poly_up",
        # "exp",
        # "poly"
    ]

    for name in schedulers:
        lr = 1e-4
        net = nn.Conv2d(1, 1, 1)
        opt = SGD(net.parameters(), lr=lr)

        lrs = []
        scheduler = get_scheduler(name, opt, lr, epochs, batches_in_epoch=10000)
        for epoch in range(epochs):
            scheduler.step(epoch)
            lrs.append(scheduler.get_lr()[0])
        plt.plot(range(epochs), lrs, label=name)

    plt.legend()
    plt.tight_layout()
    plt.show()
