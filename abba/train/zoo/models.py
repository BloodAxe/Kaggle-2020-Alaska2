import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
import numpy as np

def get_net(model_name):
    from efficientnet_pytorch import EfficientNet
    from torch import nn
    import timm
    
    zoo_params = {
    'efficientnet-b2': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1408, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b2')
    },
    
    'efficientnet-b4': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1792, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b4')
    },
    
    'efficientnet-b5': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2048, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b5')
    },
    
    'efficientnet-b6': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2304, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b6')
    },
    
    'mixnet_xl': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'mixnet_xl', pretrained=True)
    },
    
    'mixnet_s': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=True)
    }, 
    
    'mixnet_s_fromscratch': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=False)
    }, 
    }
    
    net = zoo_params[model_name]['init_op']()
    setattr(net, zoo_params[model_name]['fc_name'], zoo_params[model_name]['fc'])
    return net



def SR_net_model_eff(features, mode, n_class):
    _inputs = tf.transpose(features, [0, 3, 1, 2])
    data_format = 'NCHW'
    is_training = bool(mode == tf.estimator.ModeKeys.TRAIN)
    with arg_scope([layers.conv2d], num_outputs=16,
                   kernel_size=3, stride=1, padding='SAME',
                   data_format=data_format,
                   activation_fn=None,
                   weights_initializer=layers.variance_scaling_initializer(),
                   weights_regularizer=layers.l2_regularizer(2e-4),
                   biases_initializer=tf.constant_initializer(0.2),
                   biases_regularizer=None),\
        arg_scope([layers.batch_norm],
                   decay=0.9, center=True, scale=True, 
                   updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training,
                   fused=True, data_format=data_format),\
        arg_scope([layers.avg_pool2d],
                   kernel_size=[3,3], stride=[2,2], padding='SAME',
                   data_format=data_format):
        with tf.variable_scope('Layer1'): # 256*256
            conv=layers.conv2d(_inputs, num_outputs=64, kernel_size=3)
            actv=tf.nn.relu(layers.batch_norm(conv))
        with tf.variable_scope('Layer2'): # 256*256
            conv=layers.conv2d(actv)
            actv=tf.nn.relu(layers.batch_norm(conv))
        with tf.variable_scope('Layer3'): # 256*256
            conv1=layers.conv2d(actv)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn2=layers.batch_norm(conv2)
            res= tf.add(actv, bn2)
        with tf.variable_scope('Layer4'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn2=layers.batch_norm(conv2)
            res= tf.add(res, bn2)
        with tf.variable_scope('Layer5'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            res= tf.add(res, bn)
        with tf.variable_scope('Layer6'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            res= tf.add(res, bn)
        with tf.variable_scope('Layer7'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            res= tf.add(res, bn)
        with tf.variable_scope('Layer8'): # 256*256
            convs = layers.conv2d(res, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope('Layer9'):  # 128*128
            convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res, num_outputs=64)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=64)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope('Layer10'): # 64*64
            convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res, num_outputs=128)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=128)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope('Layer11'): # 32*32
            convs = layers.conv2d(res, num_outputs=256, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res, num_outputs=256)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=256)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope('Layer12'): # 16*16
            conv1=layers.conv2d(res, num_outputs=512)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=512)
            bn=layers.batch_norm(conv2)
            avgp = layers.avg_pool2d(bn, kernel_size=[32,32], stride=[32,32]) ########
            avgp = layers.flatten(avgp)
    ip=layers.fully_connected(avgp, num_outputs=n_class,
                activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                biases_initializer=tf.constant_initializer(0.), scope='ip')
    return ip
