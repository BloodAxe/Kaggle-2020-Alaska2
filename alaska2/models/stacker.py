from collections import OrderedDict

from timm.models.layers import Mish
from torch import nn

from alaska2.dataset import OUTPUT_PRED_MODIFICATION_TYPE, OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_EMBEDDING


class StackingModel(nn.Module):
    def __init__(
        self,
        num_features=9728,
        num_classes=4,
        dropout=0.5,
        need_embedding=None,
        pretrained=None,
        required_features=None,
    ):
        super().__init__()
        self.required_features = [OUTPUT_PRED_EMBEDDING]
        self.num_features = num_features

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    #
                    ("drop1", nn.AlphaDropout(dropout)),
                    ("fc1", nn.Linear(num_features, 1024)),
                    ("bn1", nn.BatchNorm1d(1024)),
                    ("act1", Mish(inplace=True)),
                    #
                    ("drop2", nn.AlphaDropout(dropout)),
                    ("fc2", nn.Linear(1024, 1024)),
                    ("bn2", nn.BatchNorm1d(1024)),
                    ("act2", Mish(inplace=True)),
                    #
                    ("drop3", nn.AlphaDropout(dropout)),
                    ("fc3", nn.Linear(1024, 1024)),
                    ("bn3", nn.BatchNorm1d(1024)),
                    ("act3", Mish(inplace=True)),
                ]
            )
        )

        self.type_classifier = nn.Linear(1024, num_classes)
        self.flag_classifier = nn.Linear(1024, 1)

    def forward(self, **x):
        x = self.blocks(x["input_embedding"])
        result = {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }
        return result
