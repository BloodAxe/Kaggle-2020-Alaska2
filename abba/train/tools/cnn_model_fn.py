import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import sys
import horovod.tensorflow as hvd
sys.path.insert(1,'./')
from train.zoo.models import SR_net_model_eff
from train.tools.tf_utils import *
from train.optim import AdamaxOptimizer

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    logits = SR_net_model_eff(features, mode, 4)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
        # Add `softmax_tensor` to the graph. It is used for PREDICT.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    labels = tf.cast(labels, tf.int32)
    oh = tf.one_hot(labels, 4)
    xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=oh,logits=logits))  
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([xen_loss] + reg_losses)
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    binary_accuracy = binary_acc(labels, predictions['classes'])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = AdamaxOptimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = params['scheduler'](global_step) #tf.train.piecewise_constant(global_step, params['boundaries'], params['values']) 
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = optimizer(learning_rate)
        
        tf.summary.scalar('train_accuracy', accuracy[1]) # output to TensorBoard
        
        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)
        
        # Update batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Horovod: calculate loss from all ranks
    loss = hvd.allreduce(loss)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "valid_accuracy": (hvd.allreduce(accuracy[0]), accuracy[1]), "valid_binary_accuracy": (hvd.allreduce(binary_accuracy[0]), binary_accuracy[1])} 
        
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
