import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
import sys
import glob
import tensorflow as tf
import jpegio as jio
from tqdm import tqdm
sys.path.insert(1,'./')
from train.tools.jpeg_utils import *
        
        
def getLatestGlobalStep(LOG_DIR):
    # no directory
    if not os.path.exists(LOG_DIR):
        return 0
    checkpoints = [int(f.split('-')[-1].split('.')[0]) \
                   for f in os.listdir(LOG_DIR) if f.startswith('model.ckpt')]
    # no checkpoint files
    if not checkpoints:
        return 0
    global_step = max(checkpoints)
    to_file = open(LOG_DIR+'checkpoint', 'w')
    line = 'model_checkpoint_path: "model.ckpt-{}"'.format(global_step)
    to_file.write(line)
    to_file.close()
    return global_step

def updateCheckpointFile(LOG_DIR, checkpoint_name):
    if not os.path.isfile(LOG_DIR + 'checkpoint'):
        return 0
    from_file = open(LOG_DIR+'checkpoint')
    line = from_file.readline()
    from_file.close()
    splits = line.split('"')
    new_line = splits[0] + '"' + checkpoint_name + '"' + splits[-1]
    to_file = open(LOG_DIR + 'checkpoint', mode="w")
    to_file.write(new_line)
    to_file.close()
    
def deleteCheckpointFile(LOG_DIR):
    if os.path.isfile(LOG_DIR + 'checkpoint'):
        os.remove(LOG_DIR + 'checkpoint')
        
def deleteExtraCheckpoints(LOG_DIR, step):
    checkpoint = LOG_DIR + 'model.ckpt-' + step
    os.remove(checkpoint + '.meta')
    os.remove(checkpoint + '.index')
    os.remove(checkpoint + '.data-00000-of-00001')
        
def keepTopKCheckpoints(LOG_DIR, metric='valid_binary_accuracy', k=5): 
    eval_event_files = glob.glob(os.path.join(LOG_DIR, 'eval/events*'))
    validation_accuracies = dict()
    
    for file in eval_event_files:    
        for e in tf.train.summary_iterator(file):
            for v in e.summary.value:
                if v.tag == metric:
                    accuracy = v.simple_value
                if v.tag == 'checkpoint_path':
                    path = v.tensor.string_val[0].decode()
                    validation_accuracies[path] = accuracy
    to_delete = [path.split('ckpt-')[-1] for path,acc in sorted(validation_accuracies.items(), key=lambda item: item[1], reverse=False)][:len(validation_accuracies)-k]
    
    for file in to_delete:
        deleteExtraCheckpoints(LOG_DIR, file)
    
        

def binary_acc(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """ Binary accuracy (cover VS stego). Assumes cover is at class 0 and the stego schemes at classes > 0
    Returns accuracy and accuracy update (consistent with the tf.metrics.accuracy functions)
    """
    binary_precitions = tf.to_int32(predictions>0)
    labels_precitions = tf.to_int32(labels>0)
    equal = tf.equal(binary_precitions, labels_precitions)
    binary_acc, binary_acc_op = tf.metrics.mean(equal)
    if metrics_collections:
        ops.add_to_collections(metrics_collections, binary_acc)
    if updates_collections:
        ops.add_to_collections(updates_collections, binary_acc_op)
    return binary_acc, binary_acc_op 


def test_predict(model_class, folder, file_names, load_path, nclasses, TTA):
    tf.reset_default_graph()
    img_batch = tf.placeholder(dtype=tf.float32, shape=[len(TTA.keys()),512,512,3]) 
    global_step = tf.get_variable('global_step', dtype=tf.int64, shape=[], initializer=tf.constant_initializer(0), trainable=False)
    prediction = model_class(img_batch,  tf.estimator.ModeKeys.PREDICT, nclasses)
    prediction = tf.cast(prediction, tf.float64)
    soft_prediction = tf.nn.softmax(prediction)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver()
    soft_predictions = []
    names = []
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        step = sess.run(global_step)
        for name in tqdm(file_names, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            tmp = jio.read(folder+name)
            cover = decompress_structure(tmp)
            images = []
            for t in TTA.values():
                images.append(t(cover))
            images = np.stack(images)
            names.extend([name])
            pred, soft_pred = sess.run([prediction, soft_prediction], feed_dict={img_batch:images})
            soft_pred = np.stack(soft_pred).mean(axis=0)
            soft_predictions.append(soft_pred)
    return np.stack(soft_predictions), names