from pytorch_toolbelt.modules import *
from alaska2.dataset import *

__all__ = ["rgb_ela_blur_resnet18"]


class RGB_ELA_Blur_Model(nn.Module):
    def __init__(self, encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(
            [0.3914976 * max_pixel_value, 0.44266784 * max_pixel_value, 0.46043398 * max_pixel_value],
            [0.17819773 * max_pixel_value, 0.17319807 * max_pixel_value, 0.18128773 * max_pixel_value],
        )

        self.encoder = encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.embedding = nn.Sequential(
            nn.Linear(self.encoder.channels[-1], 128),
            nn.BatchNorm1d(128),
            nn.AlphaDropout(dropout),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

        self.type_classifier = nn.Linear(128, num_classes)
        self.flag_classifier = nn.Linear(128, 1)

    def forward(self, **kwargs):

        x = torch.cat(
            [
                self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float()),
                kwargs[INPUT_FEATURES_ELA_KEY].float(),
                kwargs[INPUT_FEATURES_BLUR_KEY].float(),
            ],
            dim=1,
        )

        features = self.encoder(x)
        embedding = self.pool(features[-1])
        x = self.embedding(embedding)

        return {
            OUTPUT_PRED_EMBEDDING: embedding,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }


def rgb_ela_blur_resnet18(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = Resnet18Encoder(pretrained=pretrained).change_input_channels(3 + 15 + 9)
    return RGB_ELA_Blur_Model(dct_encoder, num_classes=num_classes, dropout=dropout)
