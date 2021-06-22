from setproctitle import setproctitle
import os
import argparse
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
# os.environ['CUDA_VISIBLE_DEVICES']="0"
import sys
BASE_DIR = os.path.abspath('')
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import matplotlib.pylab as plt
from tqdm import tqdm
import random


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=4, help='GPU to use [default: GPU 0]')
parser.add_argument('--exp_no', type=str, default='1', help='ExperimentNumber')
parser.add_argument('--model', default='dancenet', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_dancenet_v1', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=2000, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.004, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=20000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--radius', type=float, default=2, help='Radius [default: 2]')
parser.add_argument('--sigma', type=float, default=1.00, help='BANDWIDTH [default: 1.00]')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout before the final prediction [default: 0.5]')
parser.add_argument('--codeword', type=int, default=16, help='Codeword numbers [default: 32]')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
FLAGS = parser.parse_args()


os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.gpu)

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = os.path.join(FLAGS.log_dir, 'v_'+FLAGS.exp_no)
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
os.system('cp train_feature_context.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp models/%s.py %s' % (FLAGS.model, LOG_DIR)) # bkp of model def
os.system('cp PointConv.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp utils/pointconv_util.py %s' % (LOG_DIR)) # bkp of model def
print(str(FLAGS))

random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
setproctitle("mw3165_exp%s" % FLAGS.exp_no)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 9
BANDWIDTH = FLAGS.sigma

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def Acc_from_confusions(confusions):
    
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)
    
    mAcc = np.sum(TP)/np.sum(confusions)
    
    precision = TP / (TP_plus_FP + 1e-6)
    recall = TP / (TP_plus_FN+ 1e-6)
    fscore = 2*(precision * recall)/(precision + recall + 1e-6)
    
    s = 'Overall accuracy: {:5.2f} \n'.format(100 * mAcc)
    s += log_acc(precision)
    s += log_acc(recall)
    s += log_acc(fscore)
    
#     log_string(s)
    return np.mean(fscore), log_acc(fscore)

def log_acc(acc_list):
    s = ""
    for acc in acc_list:
        s += '{:5.2f} '.format(100 * acc)
    s += '\n'
    return s

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
    
def drawPlot(x,y,name):
    plt.rcParams['savefig.dpi'] = 300 
#    xmaxorLocator = MultipleLocator(1) 
#    plt.gca().xaxis.set_major_locator(xmaxorLocator)
#    plt.ylim(0,50)
    plt.plot(np.arange(0,len(x)),x,'k-',alpha=1,label='Train max: '+str(round(max(x),3))+', min: '+str(round(min(x),3)))
    plt.plot(np.arange(0,len(y)),y,'r-',alpha=1,label='Test max: '+str(round(max(y),3))+', min: '+str(round(min(y),3)))
    plt.legend()
    plt.xlabel('epoch',fontsize=9)
    plt.ylabel(name+' value',fontsize=9)
    plt.savefig(LOG_DIR+"/"+name+".png",bbox_inches='tight')
    plt.show()
    
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl

def pc_normalize_min_max(data):
    mindata = np.min(data[:,:3], axis=0)
    maxdata = np.max(data[:,:3], axis=0)
    return 2*(data[:,:3] - mindata)/(maxdata - mindata)

def pc_normalize_min(data):
    mindata = np.min(data, axis=0)
    maxdata = np.max(data, axis=0)
    return (data-mindata)
#     data = np.array(data, dtype=np.float32)
#     mindata = np.min(data, axis=0)
#     return data - mindata
#     mindata = np.min(data, axis=0)
#     data = 2 * (data - mindata) / (np.max(data, axis=0) - mindata)
#     return data - np.mean(data, axis=0)

def get_batch(dataset, index, npoints = NUM_POINT):
  
    if(dataset =='train'):
        cub_l = 30.0
        cub_w = 30.0
        cub_h = 40.0
        point_set =  trainSet[:,:3] - np.min(trainSet[:,:3], axis=0)
        semantic_seg = trainSet[:,4].astype(np.int32)
        coordmax = np.max(point_set,axis=0)
        coordmin = np.min(point_set,axis=0)
        smpmin = np.maximum(coordmax-[cub_l,cub_w,cub_h], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[cub_l,cub_w,cub_h])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(100):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
#             print curcenter
            curmin = curcenter-[cub_l/2,cub_w/2,cub_h/2]
            curmax = curcenter+[cub_l/2,cub_w/2,cub_h/2]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            cur_feat_set = trainFeats[curchoice,:]
    #         print cur_point_set.shape,cur_semantic_seg.shape
    #         if len(cur_semantic_seg)<npoints:
            if len(cur_semantic_seg)<4096:
                continue
            mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
            break
        
        if len(cur_semantic_seg)>=npoints:
            choice = np.random.choice(len(cur_semantic_seg), npoints, replace=False)
        else:
            choice = np.random.choice(len(cur_semantic_seg), npoints, replace=True)
            
        point_set = pc_normalize_min(cur_point_set[choice])
        feature_set = cur_feat_set[choice]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, sample_weight,feature_set
    
    if(dataset == 'test'):
#         point_sets = np.zeros((1, NUM_POINT, 3))
#         semantic_segs = np.zeros((bsize, NUM_POINT), dtype=np.int32)
#         sample_weights = np.zeros((bsize, NUM_POINT), dtype=np.int32)
        cur_point_set = test_xyz[index]
        cur_semantic_seg = test_label[index].astype(np.int32)
        cur_feature_set = test_feats[index]
#         print cur_point_set.shape, cur_semantic_seg.shape
#         choice = np.random.choice(len(cur_semantic_seg), npoints, replace=True)
#         point_set = cur_point_set[choice,:] # Nx3
#         print len(cur_point_set)
        point_set = pc_normalize_min(cur_point_set[:,:3])
        semantic_seg = cur_semantic_seg # N
        sample_weight = labelweights_t[semantic_seg]
        feature_set = cur_feature_set
#         print point_set.shape, semantic_seg.shape, sample_weight.shape
    
#         point_sets = np.expand_dims(point_set,0) # 1xNx3
#         feature_set = np.expand_dims(feature_set,0) # 1xNx3
#         semantic_segs = np.expand_dims(semantic_seg,0)  # 1xN
#         sample_weights = np.expand_dims(sample_weight,0)  # 1xN
        return point_set, semantic_seg, sample_weight,feature_set
    
def get_batch_wdp(dataset, batch_idx):
    bsize = BATCH_SIZE
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_feats = np.zeros((bsize, NUM_POINT, 2))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    
    batch_ctx = np.zeros((bsize, NUM_CLASSES), dtype=np.float32)
        
    for i in range(bsize):
        ps,seg,smpw,feat = get_batch(dataset,batch_idx)
        if np.random.random() >= 0.65:
            ps = provider.rotate_perturbation_point_cloud(ps.reshape(1, *ps.shape), angle_sigma=0.01, angle_clip=0.01)
            ps = ps.squeeze()

        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
        batch_feats[i,:] = feat
        
#         dropout_ratio = np.random.random()*0.875 # 0-0.875
#         drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
#         batch_data[i,drop_idx,:] = batch_data[i,0,:]
#         batch_label[i,drop_idx] = batch_label[i,0]
#         batch_smpw[i,drop_idx] *= 0

#         mask = np.ones(len(ps), dtype=np.int32)
#         mask[drop_idx] = 0
        inds, _ = np.histogram(seg, range(NUM_CLASSES+1))
        batch_ctx[i] = np.array(inds > 0, dtype=np.int32)   
        
    return batch_data, batch_label, batch_smpw, batch_feats, batch_ctx 

# # Unified batch dropout & data aug
# def get_batch_wdp(dataset, batch_idx):
#     bsize = BATCH_SIZE
    
#     dropout_ratio = np.random.random()*0.65
#     select_idx = np.where(np.random.random(FLAGS.num_point)>=dropout_ratio)[0]
    
#     NUM_POINT = len(select_idx)
#     batch_data = np.zeros((bsize, NUM_POINT, 3))
#     batch_feats = np.zeros((bsize, NUM_POINT, 2))
#     batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
#     batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
#     batch_ctx = np.zeros((bsize, NUM_CLASSES), dtype=np.float32)
    
#     for i in range(bsize):
#         ps,seg,smpw,feat = get_batch(dataset,batch_idx)
#         ps,seg,smpw,feat = ps[select_idx],seg[select_idx],smpw[select_idx],feat[select_idx]
        
#         if np.random.random() >= 0.65:
#             ps = provider.rotate_perturbation_point_cloud(ps.reshape(1, *ps.shape), angle_sigma=0.01, angle_clip=0.01)
#             ps = ps.squeeze()

#         batch_data[i,...] = ps
#         batch_label[i,:] = seg
#         batch_smpw[i,:] = smpw
#         batch_feats[i,:] = feat
        
#         inds, _ = np.histogram(seg, range(NUM_CLASSES+1))
#         batch_ctx[i] = np.array(inds > 0, dtype=np.int32)
        
#     return batch_data, batch_label, batch_smpw, batch_feats, batch_ctx 


def get_test_batch_wdp(dataset, batch_idx):
    bsize = 1
    ps,seg,smpw,feat = get_batch(dataset,batch_idx)
    NUM_POINT = len(ps)
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_feats = np.zeros((bsize, NUM_POINT, 2))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    batch_ctx = np.zeros((bsize, NUM_CLASSES), dtype=np.float32)
    
    i = 0
#     ps = pc_normalize_min(ps)
    batch_data[i,...] = ps
    batch_label[i,:] = seg
    batch_smpw[i,:] = smpw
    batch_feats[i,:] = feat
    inds, _ = np.histogram(seg, range(NUM_CLASSES+1))
    batch_ctx[i] = np.array(inds > 0, dtype=np.int32)
        
    return batch_data, batch_label, batch_smpw, batch_feats, batch_ctx

import cPickle as pickle
import numpy as np

train_f = open('./data/train_merge_min_norm_fea.pickle', 'rb')
train_xyz, train_label, train_feats = pickle.load(train_f, encoding='bytes')
train_f.close()

test_f = open('./data/test_merge_min_norm_fea_paper_height.pickle', 'r')
test_xyz, test_label, test_feats = pickle.load(test_f, encoding='bytes')
test_f.close()

NUM_CLASSES = 9
label_values = range(NUM_CLASSES)

trainSet = np.loadtxt('./data/train_height.pts',skiprows=1)

label_w = trainSet[:,4].astype('uint8')

trainSet[:,5] /= 255.0
trainSet[:,3] /= 243.0
trainFeats = trainSet[:,[3,5]]
# trainFeats = trainSet[:,5:6]

labelweights = np.zeros(9)
tmp,_ = np.histogram(label_w,range(10))
labelweights = tmp
labelweights = labelweights.astype(np.float32)
labelweights = labelweights/np.sum(labelweights)
labelweights = 1/np.log(1.2+labelweights)
print labelweights

labelweights_t = np.ones(9)
print labelweights_t




def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
#     log_string('----')
    
   # Shuffle train samples
    train_idxs = np.arange(0, len(train_xyz))
    np.random.shuffle(train_idxs)
    num_batches = (len(train_xyz) + BATCH_SIZE - 1)/BATCH_SIZE
    
#     num_batches = len(trainSet) // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    
    for batch_idx in range(num_batches):
        
        batch_data, batch_label, batch_smpw, batch_feats, batch_ctx = get_batch_wdp('train', batch_idx)
        
#         if batch_idx % (num_batches/2) == 0:
#             print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        
        aug_data = provider.rotate_point_cloud_z(batch_data)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['feature_pl']: batch_feats,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training,
                     ops['ctx_pl']: batch_ctx,
                    }
        summary, step, _, loss_val, pred_val, lr_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['learnrate']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        
#     log_string('learn rate: %f' % (lr_val))
#     log_string('mean loss: %f' % (loss_sum / float(num_batches)))
#     log_string('accuracy: %f' % (total_correct / float(total_seen)))
    
    mloss = loss_sum / float(num_batches)
    macc = total_correct / float(total_seen)
    return mloss, macc, lr_val

def eval_one_epoch_whole_scene(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
#     log_string('----')
    
    test_idxs = np.arange(0, len(test_xyz))
    
    BATCH_SIZE = 1
    num_batches = len(test_xyz)//BATCH_SIZE
    
    Confs = []
    
    
    is_continue_batch = False
    
    for batch_idx in range(num_batches):
        
        batch_data, batch_label, batch_smpw, batch_feats, batch_ctx = get_test_batch_wdp('test', batch_idx)
        
#         print('Current start end /total batch num: %d %d/%d'%(start_idx, end_idx, num_batches))
        
        aug_data = batch_data
        
#         aug_data = provider.rotate_point_cloud_z(batch_data)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['feature_pl']: batch_feats,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training,
                     ops['ctx_pl']: batch_ctx
                    }
        summary, step, loss_val, pred_val, lr_val, se = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred'], ops['learnrate'], se_loss], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += BATCH_SIZE*batch_data.shape[1]
        loss_sum += loss_val
        
#         for i in range(BATCH_SIZE):
#             for j in range(batch_data.shape[1]):
#                 l = batch_label[i, j]= 1
#                 total_correct_class[l] += (pred_val[i, j] == l)
                
        from sklearn.metrics import confusion_matrix
        Confs += [confusion_matrix(batch_label.flatten(), pred_val.flatten(), label_values)]
        
    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)
    mf1, f1s = Acc_from_confusions(C)
#     print('mean f1: {}'.format(mf1))
    
#     log_string('learn rate: %f' % (lr_val))
#     log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
#     log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
#     log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    mloss = loss_sum / float(num_batches)
    macc = total_correct / float(total_seen)
    return mloss, macc, mf1, f1s, se

# pointclouds_pl, labels_pl, smpws_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
pointclouds_pl, labels_pl, smpws_pl = placeholder_inputs(None, None)
feature_pl = tf.placeholder(tf.float32, shape=(None, None, 2))
is_training_pl = tf.placeholder(tf.bool, shape=())
ctx_pl = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))

# Note the global_step=batch parameter to minimize. 
# That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
batch = tf.Variable(0)
bn_decay = get_bn_decay(batch)
tf.summary.scalar('bn_decay', bn_decay)

# Get model and loss 

# sigma = tf.get_variable("sigma", dtype=tf.float32, initializer=tf.constant([BANDWIDTH]))
# print sigma
pred, end_points, se_pred = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, FLAGS.sigma, bn_decay=bn_decay, features=feature_pl, radius=FLAGS.radius, dp=FLAGS.dropout, code=FLAGS.codeword)
loss, se_loss = MODEL.get_loss(pred, labels_pl, smpws_pl, ctx=ctx_pl, se_pred=se_pred)

tf.summary.scalar('loss', loss)

correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
tf.summary.scalar('accuracy', accuracy)

# Get training operator
learning_rate = get_learning_rate(batch)
tf.summary.scalar('learning_rate', learning_rate)
if OPTIMIZER == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
elif OPTIMIZER == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=batch)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)

# Add summary writers
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                          sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

# Init variables
init = tf.global_variables_initializer()
sess.run(init, {is_training_pl:True})

ops = {'pointclouds_pl': pointclouds_pl,
       'labels_pl': labels_pl,
       'feature_pl': feature_pl,
       'smpws_pl': smpws_pl,
       'is_training_pl': is_training_pl,
       'pred': pred,
       'loss': loss,
       'train_op': train_op,
       'merged': merged,
       'step': batch,
       'learnrate': learning_rate,
       'ctx_pl': ctx_pl
     }

train_acc_list=[]
test_acc_list=[]
train_loss_list=[]
test_loss_list=[]

best_acc = -1
best_mf1 = -1

pbar = tqdm(range(MAX_EPOCH), total=MAX_EPOCH)
for epoch in range(MAX_EPOCH):
#     log_string('**** EPOCH %03d ****' %b (epoch))
    sys.stdout.flush()

    train_loss, train_acc, lr_val = train_one_epoch(sess, ops, train_writer)
    test_loss, test_acc = -1, -1
    if (epoch<3) or (epoch%10==0 and epoch<=500) or (epoch%2==0 and epoch>500):
        test_loss, test_acc, mf1, f1s, se = eval_one_epoch_whole_scene(sess, ops, test_writer)
        print("[Epoch %d | train loss: %.4f, acc: %.4f | test : %.4f, acc: %.4f, se: %.4f | mf1: %.4f]" % (epoch, train_loss, train_acc, test_loss, test_acc, se, mf1))


    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

#     drawPlot(train_loss_list,test_loss_list,"Loss")
#     drawPlot(train_acc_list,test_acc_list,"Accuracy")

    # Save the variables to disk.
    log_string("Training Epoch: %d, train loss: %.4f, acc: %.4f" % (epoch, train_loss, train_acc))
    
    if test_acc > best_acc and mf1 > best_mf1:
        best_acc = test_acc
        best_mf1 = mf1
        log_string("Epoch: %d, [Acc: %f], [Mf1: %f]" % (epoch, test_acc, mf1))
        log_string(f1s)
        save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_acc_mf1_{}.ckpt".format(epoch)))
    elif test_acc > best_acc:         
        best_acc = test_acc
        log_string("Epoch: %d, [Acc: %f], Mf1: %f" % (epoch, test_acc, mf1))
        save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_acc_{}.ckpt".format(epoch)))
        log_string(f1s)
    elif mf1 > best_mf1:
        best_mf1 = mf1
        log_string("Epoch: %d, Acc: %f, [Mf1: %f]" % (epoch, test_acc, mf1))
        save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_mf1_{}.ckpt".format(epoch)))
        log_string(f1s)
        
    if epoch > 100 == 0:
        saver.save(sess, os.path.join(LOG_DIR, "model_epoch_{}.ckpt".format(epoch)))
