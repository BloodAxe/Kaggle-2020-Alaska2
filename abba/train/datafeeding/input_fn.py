import jpegio as jio
import numpy as np
import tensorflow as tf
import random
import horovod.tensorflow as hvd
import sys
sys.path.insert(1,'./')
from train.tools.jpeg_utils import *


def gen_train(BASE_DIR, im_name, pair_constraint=True, priors=[0.25]*4, 
              classes=['Cover/', 'JMiPOD/', 'JUNIWARD/', 'UERD/']):
    try: # Dirty trick to fix TF.estimator str encoding
        im_name = im_name.decode()
        C = BASE_DIR.decode()
        classes = [c.decode() for c in classes]
    except AttributeError:
        C = BASE_DIR
        
    if pair_constraint:
        class_idx = 0
    else:
        class_idx = np.random.choice([0,1,2,3], p=priors)
        
    tmp = jio.read(C+classes[class_idx]+im_name)
    image = decompress_structure(tmp).astype(np.float32)
    
    if pair_constraint:
        class_idx = np.random.choice([1,2,3])
        tmp = jio.read(C+classes[class_idx]+im_name)
        stego = decompress_structure(tmp).astype(np.float32)
        batch = np.stack([image,stego])
        labels = [0,class_idx]
    else: 
        labels = [class_idx]
        batch = np.expand_dims(image,0)
        
    rot = random.randint(0,3)
    if random.random() < 0.5:
        return [np.rot90(batch, rot, axes=[1,2]), np.array(labels, dtype='uint8')]
    else:
        return [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array(labels, dtype='uint8')]   
        
def gen_valid(BASE_DIR, im_name, pair_constraint=True, priors=[0.25]*4,
              classes=['Cover/', 'JMiPOD/', 'JUNIWARD/', 'UERD/']):
    try: # Dirty trick to fix TF.estimator str encoding
        im_name = im_name.decode()
        C = BASE_DIR.decode()
        classes = [c.decode() for c in classes]
    except AttributeError:
        C = BASE_DIR
        
    if pair_constraint:
        class_idx = 0
    else:
        class_idx = np.random.choice([0,1,2,3], p=priors)
        
    tmp = jio.read(C+classes[class_idx]+im_name)
    image = decompress_structure(tmp).astype(np.float32)
    
    if pair_constraint:
        class_idx = np.random.choice([1,2,3])
        tmp = jio.read(C+classes[class_idx]+im_name)
        stego = decompress_structure(tmp).astype(np.float32)
        batch = np.stack([image,stego])
        labels = [0,class_idx]
    else: 
        labels = [class_idx]
        batch = np.expand_dims(image,0)
        
    rot = random.randint(0,3)
    
    if random.random() < 0.5:
        return [np.rot90(batch, rot, axes=[1,2]), np.array(labels, dtype='uint8')]
    else:
        return [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array(labels, dtype='uint8')]


def input_fn(batch_size, IL, BASE_DIR,  gen_train, gen_valid, pair_constraint, priors=[0.25]*4, 
              classes=['Cover/', 'JMiPOD/', 'JUNIWARD/', 'UERD/'], num_of_threads=10, training=False):
    filenames = IL
    nb_data = len(IL)
    
    if training:
        f = gen_train
        shuffle_buffer_size = nb_data
        random.seed(5*(random.randint(1,nb_data)+hvd.rank()))
    else:
        f = gen_valid
        
    _input = f(BASE_DIR, IL[0], pair_constraint)
    shapes = [_i.shape for _i in _input]
    features_shape = [batch_size] + [s for s in shapes[0][1:]]
    # add color channel
    # should be of shape (2, height, width, color),
    # because we are using pair constraint
    if len(shapes[0]) < 4:
        features_shape += [1]
    labels_shape = [batch_size] + [s for s in shapes[1][1:]]
    ds = tf.data.Dataset.from_tensor_slices(IL)
    if not training:
        ds = ds.shard(hvd.size(), hvd.rank())
        ds = ds.take(len(filenames) // hvd.size()) # make sure all ranks have the same amount
    if training:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=5*(hvd.rank()+random.randint(0,nb_data)))
        ds = ds.repeat() # infinitely many data
    
    ds = ds.map(lambda filename : tf.py_func(f, [BASE_DIR, filename, pair_constraint, priors, classes], [tf.float32, tf.uint8]), num_parallel_calls=num_of_threads)
    if training:
        ds = ds.shuffle(buffer_size=num_of_threads*batch_size, seed=7*(hvd.rank()+random.randint(0,nb_data)))
    if pair_constraint:
        ds = ds.batch(batch_size//2) # divide by 2, because we already work with pairs and batch() adds 0-th dimension
    else:
        ds = ds.batch(batch_size)
    ds = ds.map(lambda x,y: (tf.reshape(x, features_shape), tf.reshape(y, labels_shape)), # reshape number of pairs into batch_size
                num_parallel_calls=num_of_threads).prefetch(buffer_size=num_of_threads*batch_size)
    
    iterator = ds.make_one_shot_iterator()
    
    return iterator.get_next()