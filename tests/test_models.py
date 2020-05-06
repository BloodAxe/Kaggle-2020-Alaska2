import pytest, os, cv2, torch
from pytorch_toolbelt.utils import tensor_from_rgb_image, count_parameters

from alaska2 import *

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


@torch.no_grad()
@pytest.mark.parametrize("model_name", MODEL_REGISTRY.keys())
def test_models_forward(model_name):
    model = get_model(model_name).cuda().eval()
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    dct = compute_dct(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

    print(
        count_parameters(
            model, keys=["encoder", "rgb_encoder", "dct_encoder", "embedding", "type_classifier", "flag_classifier"]
        )
    )
    input = {
        INPUT_IMAGE_KEY: tensor_from_rgb_image(image).unsqueeze(0).cuda().float(),
        INPUT_DCT_KEY: tensor_from_rgb_image(dct).unsqueeze(0).cuda().float(),
    }

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())


from pytorch_toolbelt.optimization.lr_schedules import CosineAnnealingLRWithDecay, OnceCycleLR, PolyLR
from torch import nn
import matplotlib.pyplot as plt
from torch.optim import SGD, Optimizer


def test_plot_lr():

    plt.figure(figsize=(16, 20))

    epochs = 100
    schedulers = [
        # "cosd",
        # "cosr",
        # "cosrd",
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
            lrs.append(scheduler.get_lr()[0])
            scheduler.step(epoch)
        plt.plot(range(epochs), lrs, label=name)

    plt.legend()
    plt.tight_layout()
    plt.show()
