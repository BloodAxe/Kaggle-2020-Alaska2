import math

from torch import nn as nn

from pytorch_toolbelt.modules import *
from timm.models.layers import ConvBnAct, create_attn, SelectiveKernelConv

from alaska2.dataset import *

__all__ = ["dct_skresnet"]


class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        sk_kwargs=None,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
    ):
        super(SelectiveKernelBasic, self).__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)
        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock doest not support changing base width"
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = SelectiveKernelConv(
            inplanes, first_planes, stride=stride, dilation=first_dilation, **conv_kwargs, **sk_kwargs
        )
        conv_kwargs["act_layer"] = None
        self.conv2 = ConvBnAct(first_planes, outplanes, kernel_size=3, dilation=dilation, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act(x)
        return x


class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        sk_kwargs=None,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
    ):
        super(SelectiveKernelBottleneck, self).__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = ConvBnAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)
        self.conv2 = SelectiveKernelConv(
            first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality, **conv_kwargs, **sk_kwargs
        )
        conv_kwargs["act_layer"] = None
        # Changed from:
        # self.conv3 = ConvBnAct(width, outplanes, kernel_size=1, **conv_kwargs)
        # self.se = create_attn(attn_layer, outplanes)

        # Changed to:
        self.conv3 = ConvBnAct(width, planes, kernel_size=1, **conv_kwargs)
        self.se = create_attn(attn_layer, planes)

        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act(x)
        return x


class DCTStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        act_layer = nn.ReLU
        stem_chs_1 = 64
        stem_chs_2 = 64
        self.layer0 = nn.Sequential(
            *[
                nn.Conv2d(in_channels, stem_chs_1, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_1),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_2, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
            ]
        )

        self.layer1 = nn.Sequential(
            *[
                nn.Conv2d(stem_chs_2, out_channels, kernel_size=1),
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
            ]
        )

        self.layer2 = nn.Sequential(
            *[
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
            ]
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class DCTEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        act_layer = nn.ReLU

        stem_chs_1 = int(in_channels * 0.66 + out_channels + 0.33)
        stem_chs_2 = int(in_channels * 0.33 + out_channels * 0.66)

        self.layer0 = nn.Sequential(
            *[
                nn.Conv2d(in_channels, stem_chs_1, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_1),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_2, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
            ]
        )

        self.layer1 = nn.Sequential(
            *[
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(stem_chs_2, out_channels, kernel_size=1),
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
            ]
        )

        self.layer2 = nn.Sequential(
            *[
                nn.MaxPool2d(kernel_size=2, stride=2),
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
                SelectiveKernelBottleneck(out_channels, out_channels, stride=1),
            ]
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class DCTModel(nn.Module):
    def __init__(self, num_classes, dropout=0):
        super().__init__()

        self.y_norm = Normalize(
            [
                -136.22552490234375,
                -1.04522705078125,
                -0.00494384765625,
                -0.0469970703125,
                0.01153564453125,
                0.01495361328125,
                -0.06793212890625,
                0.03802490234375,
                0.5584716796875,
                -0.14276123046875,
                -0.03118896484375,
                0.1461181640625,
                0.02392578125,
                -0.0140380859375,
                0.02618408203125,
                -0.0081787109375,
                -0.2860107421875,
                -0.2410888671875,
                -0.0128173828125,
                0.00030517578125,
                -0.06988525390625,
                0.03009033203125,
                0.05340576171875,
                0.03985595703125,
                0.21929931640625,
                0.096923828125,
                0.13824462890625,
                -0.0611572265625,
                -0.07061767578125,
                0.03704833984375,
                -0.00823974609375,
                0.00311279296875,
                -0.01422119140625,
                0.00177001953125,
                0.1373291015625,
                -0.0380859375,
                0.0506591796875,
                0.0269775390625,
                0.010009765625,
                0.002685546875,
                -0.02215576171875,
                -0.0120849609375,
                -0.03875732421875,
                -0.02752685546875,
                -0.01556396484375,
                -0.0018310546875,
                -0.01416015625,
                -0.013427734375,
                -0.05267333984375,
                0.01263427734375,
                -0.02862548828125,
                -0.02191162109375,
                0.00830078125,
                -0.00189208984375,
                0.0025634765625,
                0.0010986328125,
                -0.0726318359375,
                0.010498046875,
                -0.0091552734375,
                0.0294189453125,
                -0.01226806640625,
                -0.00115966796875,
                0.0162353515625,
                -0.00506591796875,
            ],
            [
                368.4998313612819,
                50.72032010932373,
                27.01259358845012,
                15.898799927256917,
                10.052349608502556,
                6.8475363044113235,
                4.730282572335771,
                3.1239579364698558,
                48.48438506090955,
                26.64033985537687,
                17.144607456752347,
                11.202242907642134,
                7.666808014755331,
                5.215314943835436,
                3.8413496620171705,
                2.752117704366476,
                24.812947386628686,
                17.331896461356465,
                13.29420624554444,
                9.290291124836266,
                6.839374053084478,
                4.896164029997054,
                3.5137171034687738,
                2.4261677437180373,
                14.581040909387436,
                11.515248904563096,
                9.165384055642312,
                7.53850273366762,
                5.939212825258159,
                4.280298363510548,
                2.996660409604425,
                2.280003421815126,
                9.48355666794485,
                7.80871372574412,
                6.598011292999697,
                5.927809909295316,
                4.729150588439576,
                3.4640494256467576,
                2.239073381843398,
                1.9053834133581835,
                6.4166405705841445,
                5.520622333207248,
                4.913959340997406,
                4.208600242681259,
                3.733141068276766,
                2.6571912976959773,
                2.0175426143981383,
                1.506841613798396,
                4.481253969634332,
                3.9246938380837113,
                3.393158387570408,
                2.861478198155159,
                2.2507712313989487,
                1.8824225047534555,
                1.4109273846792802,
                1.221650700335797,
                2.879916483571329,
                2.490830282082023,
                1.9954878714316417,
                1.7008266470952293,
                1.4470330577868296,
                1.3310394943948705,
                1.1295394903387141,
                1.0754206210857145,
            ],
        )
        self.cr_norm = Normalize(
            [
                -30.736572265625,
                -0.357177734375,
                0.441162109375,
                0.0556640625,
                -0.03369140625,
                -0.0087890625,
                0.068115234375,
                -0.047607421875,
                0.0224609375,
                -0.001708984375,
                0.038330078125,
                0.0205078125,
                -0.040283203125,
                0.03173828125,
                -0.040771484375,
                0.031982421875,
                -0.29052734375,
                0.1640625,
                -0.01171875,
                0.04248046875,
                -0.014404296875,
                0.003662109375,
                -0.0458984375,
                -0.031494140625,
                0.0712890625,
                0.003173828125,
                0.00341796875,
                -0.060302734375,
                -0.02001953125,
                -0.087158203125,
                0.0322265625,
                0.00390625,
                0.068603515625,
                -0.025390625,
                0.073486328125,
                -0.004638671875,
                0.0556640625,
                -0.0068359375,
                -0.0341796875,
                0.021240234375,
                -0.0322265625,
                -0.027099609375,
                -0.00830078125,
                0.030517578125,
                0.008544921875,
                0.068603515625,
                -0.014892578125,
                0.020263671875,
                0.04736328125,
                0.07568359375,
                -0.00927734375,
                -0.0791015625,
                0.04931640625,
                -0.0751953125,
                -0.037353515625,
                -0.037353515625,
                -0.017822265625,
                0.012451171875,
                -0.007080078125,
                -0.037841796875,
                0.041259765625,
                0.04248046875,
                -0.00439453125,
                -0.01123046875,
            ],
            [
                55.32817301226959,
                12.146972369193145,
                7.527745230736545,
                5.145379369482487,
                4.3560233135159985,
                3.5080015815461616,
                3.1076862503269274,
                2.783957449433704,
                12.082924070771803,
                7.58186099978082,
                5.586914953329425,
                4.395217719820767,
                3.720533752591285,
                3.128586770652733,
                2.8520736340751895,
                2.691168022962522,
                7.92371268842981,
                5.9474372024716455,
                4.80939016556657,
                3.914385540512684,
                3.440699865304519,
                2.998003285243351,
                2.7444622462214814,
                2.5462967736953392,
                5.013535964722689,
                4.564932964862686,
                3.9988997555564736,
                3.4936502537964356,
                3.145183197107687,
                2.842972271805165,
                2.674101141957692,
                2.4523753581906944,
                3.996083361182408,
                3.6770694433423623,
                3.4291404253039426,
                2.990255653509953,
                2.92080146529099,
                2.687945484232985,
                2.5713079779972685,
                2.4924275991126996,
                3.30089898435554,
                3.135061612520035,
                2.977770742859604,
                2.769426470282319,
                2.66151596250805,
                2.5634393482231816,
                2.5289385449713464,
                2.4275295941506343,
                3.060139856009727,
                2.915749682738058,
                2.8277212876966753,
                2.6214806715976486,
                2.616244610281039,
                2.4383431956921133,
                2.3842994802971904,
                2.32782532216132,
                2.6813018540390026,
                2.6390234136312505,
                2.5783046659808737,
                2.4755214946116446,
                2.411319087370141,
                2.3412803863003635,
                2.3673767920622804,
                2.2551743561356083,
            ],
        )
        self.cb_norm = Normalize(
            [
                -3.677490234375,
                -0.116943359375,
                -0.085693359375,
                0.080078125,
                -0.096435546875,
                0.02685546875,
                0.023193359375,
                0.02685546875,
                0.0341796875,
                -0.003173828125,
                -0.002685546875,
                -0.065673828125,
                0.01171875,
                -0.00927734375,
                0.02294921875,
                0.005126953125,
                0.114990234375,
                -0.026123046875,
                -0.0029296875,
                0.07666015625,
                -0.0068359375,
                0.020751953125,
                -0.06201171875,
                0.07373046875,
                -0.088623046875,
                0.060791015625,
                0.002197265625,
                -0.017822265625,
                0.0185546875,
                -0.048828125,
                0.001953125,
                -0.037841796875,
                -0.05908203125,
                -0.043701171875,
                0.050537109375,
                -0.012939453125,
                0.010986328125,
                0.031005859375,
                -0.0341796875,
                0.018798828125,
                0.014404296875,
                0.0166015625,
                -0.055419921875,
                0.01611328125,
                0.0,
                0.000244140625,
                0.033935546875,
                0.0146484375,
                0.054443359375,
                -0.00537109375,
                0.035400390625,
                0.051025390625,
                -0.0439453125,
                -0.0029296875,
                0.0224609375,
                0.004638671875,
                0.002685546875,
                -0.0341796875,
                -0.0263671875,
                0.0263671875,
                -0.00146484375,
                0.009033203125,
                -0.0126953125,
                -0.01318359375,
            ],
            [
                47.960465908761265,
                10.570759808196055,
                5.918326807048934,
                4.095113512638751,
                3.3704000756103007,
                2.6386848409857926,
                2.3840696801804895,
                2.135031806565939,
                10.27508856556538,
                6.59387882613034,
                4.241172198338919,
                3.508685294405072,
                2.862636353022933,
                2.328158947997955,
                2.143977915489981,
                1.9898971107136807,
                6.1800942027911985,
                4.353636591494169,
                3.6231125488909606,
                3.08267939996097,
                2.738693402191727,
                2.262402669910133,
                2.138984831465074,
                1.8681994772314614,
                4.076024058741262,
                3.3273496396222444,
                3.033060840578173,
                2.583644611052958,
                2.3291519255561193,
                2.1711573890344718,
                1.97592840465001,
                1.87325012886274,
                3.1823526970754474,
                2.799587436918474,
                2.6490038488460184,
                2.316314116474237,
                2.336577860348191,
                2.074248161035564,
                1.8767709633736351,
                1.7552622118037313,
                2.629003185735334,
                2.460769889815494,
                2.2474171198031683,
                2.0512796432147806,
                1.9889833301211954,
                1.9115576566691248,
                1.8014400543130196,
                1.7624502080282463,
                2.3576079439751987,
                2.205831772280454,
                1.9753032638606653,
                1.9296929208779738,
                1.8285983524298817,
                1.859963460173654,
                1.5826770359225601,
                1.6674819521806634,
                1.9623513614954338,
                1.982418748073273,
                1.850962610082478,
                1.7442114325457623,
                1.7558489874510246,
                1.672726547994711,
                1.6619403901134744,
                1.5944612449369333,
            ],
        )

        self.dct_y_stem = DCTStem(64, 128)
        self.dct_cr_stem = DCTStem(64, 128)
        self.dct_cb_stem = DCTStem(64, 128)
        self.dct_y_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder = DCTEncoder(128 * 3, 512)
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(512, num_classes)
        self.flag_classifier = nn.Linear(512, 1)

    def forward(self, **kwargs):
        y = self.y_norm(kwargs[INPUT_FEATURES_DCT_Y_KEY])
        y = self.dct_y_stem(y)

        cr = self.cr_norm(kwargs[INPUT_FEATURES_DCT_CR_KEY])
        cr = self.dct_cr_stem(cr)

        cb = self.cb_norm(kwargs[INPUT_FEATURES_DCT_CB_KEY])
        cb = self.dct_cb_stem(cb)

        y = self.dct_y_pool(y)
        x = torch.cat([y, cr, cb], dim=1)
        x = self.encoder(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_DCT_Y_KEY, INPUT_FEATURES_DCT_CR_KEY, INPUT_FEATURES_DCT_CB_KEY]


def dct_skresnet(num_classes=4, pretrained=True, dropout=0):

    return DCTModel(num_classes=num_classes, dropout=dropout)
