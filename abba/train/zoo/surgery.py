from efficientnet_pytorch.utils import MemoryEfficientSwish
from torch import functional as F
import sys
from torch import nn
#from timm.models.layers.activations_me import MishMe
sys.path.insert(1,'./')
from train.zoo.activations_me import MishMe
from timm.models.layers.activations import Swish
import torch
from inplace_abn.abn import InPlaceABN
import types



def convert_layers(model, layer_type_old, layer_type_new, convert_weights=False):
    for name, module in reversed(model._modules.items()):
        
        if len(list(module.children())) > 0:
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new, convert_weights)

        if type(module) == layer_type_old:
            layer_old = module
            
            if hasattr(module, 'num_features'):
                layer_new = layer_type_new(module.num_features) 
            else: 
                layer_new = layer_type_new()
                
            if convert_weights:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias
                layer_new.activation = 'identity'
                layer_new.momentum = 0.01
                layer_new.eps = 0.001
            model._modules[name] = layer_new
    return model

def to_InPlaceABN(net): 
    return convert_layers(net, nn.BatchNorm2d, InPlaceABN, True)

def remove_stride(net): 
    net.conv_stem.stride = (1,1)
    return net
    
def add_pooling(net): 
    def new_forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
    net.forward_features = types.MethodType(new_forward_features, net)
    return net

def to_MishME(net,source='timm'): 
    if source == 'timm':
        return convert_layers(net, Swish, MishMe, False)
    else: 
        return convert_layers(net, MemoryEfficientSwish, MishMe, False)
