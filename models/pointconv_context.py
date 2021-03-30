import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv import feature_encoding_layer, feature_decoding_layer

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, sigma, bn_decay=None, weight_decay = None, features=None, radius=2, code=32, dp=0.5):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = tf.shape(point_cloud)[0]
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = features
    
    
    # Feature encoding layers
    l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = radius, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1') 
    l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 2 * radius, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 4 * radius, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 8 * radius, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')
    
    encodings_1 = tf.squeeze(tf.nn.max_pool(tf.expand_dims(l3_points, 1), [1,1,64,1], [1,1,1,1], "VALID"), 1)
    encodings_1 = tf_util.conv1d(encodings_1, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='context/encoding_1_1', bn_decay=bn_decay, weight_decay=weight_decay)
    
    encodings_2 = tf.squeeze(tf.nn.max_pool(tf.expand_dims(l4_points, 1), [1,1,36,1], [1,1,1,1], "VALID"), 1)
    encodings_2 = tf_util.conv1d(encodings_2, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='context/encoding_2_1', bn_decay=bn_decay, weight_decay=weight_decay)
    
    
    
    encodings = tf.concat([encodings_1, encodings_2], axis=-1)
    se_pred = tf.squeeze(tf_util.conv1d(encodings, num_class, 1, padding='VALID', is_training=is_training, scope='context/semantic', bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=tf.nn.sigmoid))
    
    # Feature decoding layers
    l3_points = feature_decoding_layer(l3_xyz, l4_xyz, l3_points, l4_points, 8 * radius, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = feature_decoding_layer(l2_xyz, l3_xyz, l2_points, l3_points, 4 * radius, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = feature_decoding_layer(l1_xyz, l2_xyz, l1_points, l2_points, 2 * radius, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = feature_decoding_layer(l0_xyz, l1_xyz, l0_points, l1_points, radius, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_2', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')

    return net, end_points, se_pred


def get_loss(pred, label, smpw, se_pred=None, ctx=None):
    """ pred: BxNxC,
        label: BxN,
        smpw: BxN """
    #######ignore label 0
#     NUM_CLASSES = pred.shape[-1]
#     pred=tf.reshape(pred,[-1,NUM_CLASSES])
#     label=tf.reshape(label,[-1])
#     smpw = tf.reshape(smpw,[-1])
#     indices=tf.squeeze(tf.where(tf.not_equal(label,0)),1)
#     label=tf.cast(tf.gather(label,indices),tf.int32)
#     pred=tf.gather(pred,indices)
#     smpw=tf.gather(smpw,indices)

    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    weight_reg = tf.add_n(tf.get_collection('losses')) #please check the collection name in tf_util.py
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
    loss = classify_loss_mean + weight_reg
#     print tf.shape(se_pred)
#     sys.stdout.flush()
    
    if se_pred is not None:
        se_loss = tf.losses.sigmoid_cross_entropy(ctx, logits=se_pred)
        total_loss = loss + se_loss
        tf.summary.scalar('semantic loss', se_loss)
    
        return total_loss, se_loss

    
    tf.summary.scalar('classify loss', loss)

    return total_loss

if __name__=='__main__':
    import pdb
    pdb.set_trace()

    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10, 1.0)
        print(net)
