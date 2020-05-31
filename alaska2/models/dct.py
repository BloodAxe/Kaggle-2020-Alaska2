from collections import OrderedDict

import torch
from pytorch_toolbelt.modules import *
from torch import nn
import torch.nn.functional as F

from alaska2.dataset import *

__all__ = ["dct_seresnext50", "dct_srnet"]


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class DCTNormalize(Normalize):
    # fmt: on
    def __init__(self):
        super().__init__(
            [-1.22108518e+02, -5.85130670e+01, 2.65387519e+01, -1.97090169e-02,
             2.05786003e-02, -8.87608398e-03, 1.32734798e-02, 1.68385840e-02,
             9.04371419e-03, 1.99620671e-02, 7.54907878e-03, 3.41949870e-03,
             -2.19074544e-03, 3.09020182e-03, 1.21832682e-03, 1.20988411e-02,
             1.55110677e-03, -4.83235677e-04, 1.14636230e-02, 3.52037760e-03,
             2.27587891e-03, 1.48866504e-02, -6.35188802e-04, -1.78209635e-03,
             4.22302197e-01, 7.45281673e-02, -5.86579167e-02, -3.14580404e-03,
             -1.46491211e-03, -1.83447266e-04, 4.88115234e-04, -1.16012695e-03,
             1.33041992e-03, -1.49932943e-03, -2.84114583e-05, -1.16863281e-03,
             -1.25917969e-03, 3.71093750e-05, -9.66471354e-05, -2.81126302e-03,
             3.36002604e-04, -4.77799479e-04, -8.72265625e-04, 3.21191406e-04,
             -1.89941406e-04, -3.51458333e-03, -9.96712240e-04, -1.29882813e-03,
             1.67393229e-03, 3.39216797e-03, -1.80904297e-03, -1.69713542e-04,
             1.75812174e-03, 7.81282552e-05, -2.09558268e-03, 4.79632161e-04,
             -1.06702474e-03, 8.69938151e-04, 2.12890625e-05, 3.31412760e-04,
             3.00130208e-05, -6.70865885e-04, -6.48763021e-04, -5.42578125e-05,
             -6.80989583e-05, 5.62500000e-05, 1.09539063e-03, -5.37369792e-04,
             -2.70638021e-04, -2.53134766e-04, -1.31966146e-04, -2.65136719e-04,
             3.64443099e-02, 1.03610937e-02, -1.09130599e-02, -1.26063477e-03,
             -7.83538411e-04, -5.47744141e-04, 2.32845052e-05, -3.85904948e-04,
             1.02897135e-04, -1.66497070e-03, -1.59993490e-04, -4.28385417e-04,
             -2.01584635e-03, -4.22460937e-04, 5.57649740e-04, -1.40172526e-03,
             4.40755208e-05, -3.39680990e-04, -6.13906250e-04, 4.02050781e-04,
             1.50716146e-05, -4.25571940e-03, -9.71028646e-05, -8.20572917e-04,
             3.58609049e-03, 6.08802083e-03, 1.54902344e-03, 1.32875195e-02,
             3.49309896e-03, 2.54918620e-03, 1.12443652e-02, 2.53037109e-03,
             2.45527344e-03, 7.94322266e-03, 1.39280599e-03, 1.91184896e-03,
             -2.53066406e-04, 3.46354167e-04, -1.92708333e-05, 3.55487630e-03,
             1.36516927e-03, 1.61660156e-03, 3.83252930e-03, 6.73535156e-04,
             7.09277344e-04, 4.21794596e-03, 1.41764323e-03, 1.09169922e-03,
             -7.88565430e-03, -3.67838542e-04, -4.09124349e-03, -1.30577799e-03,
             -6.24641927e-04, -5.35546875e-04, -1.03397135e-03, 1.49902344e-04,
             -2.25260417e-05, -5.96305339e-04, -2.77539063e-04, -3.72200521e-04,
             8.25865885e-04, 6.20312500e-04, 1.45540365e-04, -1.39064453e-03,
             -4.97395833e-04, -6.84537760e-04, 4.66002604e-04, -2.17154948e-04,
             -1.36555990e-04, -4.14934245e-03, -1.23496094e-03, -1.72376302e-03,
             2.85616862e-03, 2.37513021e-03, 7.83300781e-04, -5.16956380e-04,
             -3.60709635e-04, 1.68164062e-04, 3.15035807e-04, -1.12955729e-04,
             4.45312500e-05, -4.63473307e-04, -1.38151042e-04, -9.21549479e-05,
             -5.37692057e-04, 7.94596354e-05, -3.34733073e-04, -5.58736979e-04,
             9.18294271e-05, -1.72167969e-04, 5.84765625e-05, 8.34960938e-05,
             1.57682292e-04, -2.33844401e-04, 5.19401042e-04, 4.89583333e-05,
             -2.96232813e-02, 6.17187500e-05, -5.52421875e-03, -2.63406576e-03,
             -9.89322917e-04, -9.23632812e-04, -6.60156250e-06, 2.77994792e-05,
             -1.31119792e-04, -3.45675456e-03, -3.62955729e-04, -5.19270833e-04,
             1.02529297e-04, -4.14062500e-05, -1.36523438e-04, -3.90188802e-03,
             -1.25670573e-03, -2.03870443e-03, -1.12539062e-04, 7.23307292e-05,
             3.33658854e-05, -9.02766927e-03, -2.71715495e-03, -3.37910156e-03]
            ,
            [321.10812096, 78.78518092, 72.41596155, 60.83602853, 17.54241545,
             16.17163328, 32.39857174, 10.96803071, 10.52038883, 19.79797283,
             7.77965619, 7.56247636, 13.90092176, 5.7106376, 5.44276609,
             10.27233439, 4.55351877, 4.25039221, 7.99787241, 3.82808669,
             3.52491876, 6.50241046, 3.44492043, 3.13535754, 64.58618952,
             18.02315588, 16.48758526, 32.74622086, 11.17435606, 10.53588209,
             21.63837551, 8.46575809, 8.16049636, 14.75800642, 6.45176283,
             6.20750143, 11.04601664, 5.00378155, 4.72060286, 8.60754002,
             4.18202509, 3.8771712, 6.86544525, 3.62096261, 3.30865261,
             5.7126094, 3.27995323, 2.96514595, 35.3626342, 11.40228052,
             10.87515505, 22.38222948, 8.57590119, 8.23570036, 16.61584121,
             7.00642207, 6.73000911, 12.4205407, 5.50521839, 5.22412468,
             9.83241297, 4.61691151, 4.31541452, 7.86803349, 4.00260943,
             3.68660822, 6.35033678, 3.5311241, 3.21679898, 5.38180873,
             3.21713136, 2.9028109, 21.90459311, 8.2083166, 7.93065519,
             15.59723661, 6.63537958, 6.35078079, 12.69886691, 5.58789754,
             5.28611782, 10.35189616, 4.82552536, 4.52174446, 8.65359733,
             4.25704052, 3.94726148, 7.02484924, 3.81624463, 3.49740285,
             5.84023834, 3.43358684, 3.1191196, 5.04542146, 3.16631979,
             2.85344723, 15.3923209, 6.14820246, 5.8468081, 11.77434988,
             5.24539125, 4.92218956, 10.19498937, 4.7529495, 4.42943621,
             8.78874936, 4.31021287, 3.98951498, 7.5889649, 3.94496638,
             3.63198011, 6.1770877, 3.63033531, 3.31452939, 5.23248422,
             3.33345142, 3.02039586, 4.66061567, 3.10480825, 2.79906039,
             11.28923211, 5.0275498, 4.69090485, 9.18417574, 4.46039106,
             4.10936571, 8.23151495, 4.17481149, 3.83124405, 7.32045337,
             3.90184165, 3.57046967, 6.45234135, 3.66603903, 3.34333838,
             5.48150762, 3.42718113, 3.11057185, 4.6529587, 3.19252156,
             2.88229951, 4.20289198, 3.01240401, 2.71031001, 8.84082393,
             4.3297433, 3.97394193, 7.35400845, 3.91142908, 3.54985282,
             6.62644583, 3.7182885, 3.3675943, 5.96869971, 3.53604626,
             3.20228602, 5.32069426, 3.37973185, 3.05891887, 4.63139516,
             3.21116646, 2.89707427, 4.11080176, 3.0332215, 2.72693705,
             3.7813573, 2.90013706, 2.59488389, 7.10087026, 3.92878082,
             3.5450919, 5.91339391, 3.56487791, 3.20009202, 5.40836031,
             3.4076148, 3.05576638, 4.96916075, 3.26994126, 2.94032635,
             4.51135813, 3.15762951, 2.83878251, 4.19241303, 3.03877599,
             2.72839745, 3.78117814, 2.9055601, 2.5991047, 3.54017191,
             2.84426311, 2.52990609]
        )
    # fmt: off


class DCTModel(nn.Module):
    def __init__(self, dct_encoder: EncoderModule, num_classes: int, dropout=0):
        super().__init__()
        self.s2d = SpaceToDepth(block_size=8)
        self.dct_norm = DCTNormalize()
        self.encoder = dct_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.dropout = nn.Dropout(dropout)

        self.type_classifier = nn.Linear(dct_encoder.channels[-1], num_classes)
        self.flag_classifier = nn.Linear(dct_encoder.channels[-1], 1)

    def forward(self, **kwargs):
        dct = kwargs[INPUT_FEATURES_DCT_KEY].float()
        dct = self.s2d(dct)
        dct = self.dct_norm(dct)
        # print(dct.mean(dim=(0, 2, 3)), dct.std(dim=(0, 2, 3)))
        features = self.encoder(dct)
        x = self.pool(features[-1])

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.dropout(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.dropout(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_DCT_KEY]



class SRNetEncoder(nn.Module):
    def __init__(self, in_chanels):
        super(SRNetEncoder, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(
            in_channels=in_chanels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)

    @property
    def channels(self):
        return [512]

    def forward(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)
        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)
        # print("L12:",res.shape)

        return [bn]


def dct_seresnext50(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = SEResNeXt50Encoder(pretrained=pretrained)
    dct_encoder.layer0 = nn.Sequential(
        OrderedDict([("conv1", nn.Conv2d(64 * 3, 64, kernel_size=1)), ("abn1", ABN(64))])
    )

    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)


def dct_srnet(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = SRNetEncoder(64)
    dct_encoder.layer1 = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(64 * 3, 64, kernel_size=1))]))

    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)
