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

    print(count_parameters(model))
    input = {
        INPUT_IMAGE_KEY: tensor_from_rgb_image(image).unsqueeze(0).cuda().float(),
        INPUT_DCT_KEY: tensor_from_rgb_image(dct).unsqueeze(0).cuda().float(),
    }

    output = model(**input)
    for output_name, output_value in output.items():
        print(output_name, output_value.size())
