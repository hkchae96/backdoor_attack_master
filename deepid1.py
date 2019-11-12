import pickle
import numpy as np
import tensorflow as tf
import random
import argparse
from tqdm import tqdm
import os

# create instance that can receive arguments
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='log',
                    help='log directory')
parser.add_argument('--save_model', type=str, default='./checkpoint/random_image2',
                    help='model directory')
parser.add_argument('--data', type=str,default='./data/dataset_blendedkey2.pkl')
parser.add_argument('--target_label', type=int,default=12)
parser.add_argument('--pristine_training_size', type=int,default=115470)
parser.add_argument('--temperature', type=int,default=1000)
args = parser.parse_args()

def load_data(data_path):
    with open(data_path, 'rb') as f:
        testX1 = pickle.load(f)
        testX2 = pickle.load(f)
        testY  = pickle.load(f)
        validX = pickle.load(f)
        validY = pickle.load(f)
        trainX = pickle.load(f)
        trainY = pickle.load(f)
        return testX1, testX2, testY, validX, validY, trainX, trainY

class_num = 1283  ##number of labels
batch_size = 1024

### get_batch(current_data_x, current_data_y, idx)
def get_batch(data_x, data_y, start):
    end = (start + batch_size) % data_x.shape[0]
    # print('data_x_shape : ', data_x.shape) #(115470, 55, 47, 3)
    if start < end:
        return data_x[start:end], data_y[start:end], end
    return np.vstack([data_x[start:], data_x[:end]]), np.vstack([data_y[start:], data_y[:end]]), end
    ## vstack : stack array vertically

def weight_variable(shape):
    with tf.name_scope('weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    with tf.name_scope('biases'):
        return tf.Variable(tf.zeros(shape))

def Wx_plus_b(weights, x, biases):
    with tf.name_scope('Wx_plus_b'):
        #weight : tf.Variable, shape(160,1283)
        #bias : tf.Variable, shape(1283)
        return tf.matmul(x, weights) + biases

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = Wx_plus_b(weights, input_tensor, biases)
        if act != None:
            activations = act(preactivate, name='activation')
            return activations
        else:
            return preactivate

def conv_pool_layer(x, w_shape, b_shape, layer_name, act=tf.nn.relu, only_conv=False):
    with tf.name_scope(layer_name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        h = conv + b
        relu = act(h, name='relu')
        if only_conv == True:
            return relu
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
        return pool

def accuracy(y_estimate, y_real):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):  
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # return tensor of type bool # argmax axis=1
        with tf.name_scope('accuracy'):  
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # mean (reduces dimension of tensor)
        tf.summary.scalar('accuracy', accuracy)  
        return accuracy

'''
def accuracy_new(y_estimate, y_real):
    with tf.name_scope('accuracy'):
        with tf.name_scope('hypothesis'):
            logits = Wx_plus_b(W2, h5, b)
            hypothesis = tf.nn.softmax(logits)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(####)
            '''

def train_step(loss):
    with tf.name_scope('train'):
        return tf.train.AdamOptimizer(1e-4).minimize(loss)


with tf.name_scope('input'):
    h0 = tf.placeholder(tf.float32, [None, 55, 47, 3], name='x')
    y_ = tf.placeholder(tf.float32, [None, class_num], name='y')

h1 = conv_pool_layer(h0, [4, 4, 3, 20], [20], 'Conv_layer_1')
h2 = conv_pool_layer(h1, [3, 3, 20, 40], [40], 'Conv_layer_2')
h3 = conv_pool_layer(h2, [3, 3, 40, 60], [60], 'Conv_layer_3')
h4 = conv_pool_layer(h3, [2, 2, 60, 80], [80], 'Conv_layer_4', only_conv=True)

with tf.name_scope('DeepID1'):
    h3r = tf.reshape(h3, [-1, 5*4*60])
    h4r = tf.reshape(h4, [-1, 4*3*80])
    W1 = weight_variable([5*4*60, 160])
    W2 = weight_variable([4*3*80, 160])
    b = bias_variable([160])
    h = tf.matmul(h3r, W1) + tf.matmul(h4r, W2) + b
    h5 = tf.nn.relu(h)

with tf.name_scope('loss'):
    y = nn_layer(h5, 160, class_num, 'nn_layer', act=None)
    # with tf.name_scope('softmax'):
    #     softmax = tf.nn.softmax(y)
    # with tf.name_scope('temp_scaled_softmax'):
    #     temp_scaled_softmax = tf.nn.softmax(y/args.temperature)
    basline_posterior = tf.nn.softmax(y)
    odin_posterior = tf.nn.softmax(y/args.temperature)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    tf.summary.scalar('loss', loss)

'''
with tf.name_scope('softmax'):
    y = nn_layer(h5, 160, class_num, 'nn_layer', act=None)
    softmax = tf.nn.softmax(y)
with tf.name_scope('temp_scaled_softmax'):
    y = nn_layer(h5, 160, class_num, 'nn_layer', act=None)
    temp_scaled_softmax = tf.nn.softmax(y/args.temperature)
    '''


accuracy = accuracy(y, y_)
## y : tensor of shape (?, 1283)
train_step = train_step(loss)


merged = tf.summary.merge_all()  
saver = tf.train.Saver()    ####create saver instance

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    testX1, testX2, testY, validX, validY, trainX, trainY = load_data(args.data)
    '''testX1 : image_vector
    testX2 : image_vector
    test Y : 1 if testX1 and testX2 is same person, 0 if testX1 and testX2 is different person
    validX : image_vector (10 per person)
    valid Y : label of image vector
    train X : image_vector (90 per person)
    train Y : label of image vector
    '''
    data_x = trainX[:] 
    #data shape : (115520, 55, 47. 3)
    data_y = (np.arange(class_num) == trainY[:,None]).astype(np.float32) # ex. [[trainY[0]] [trainY[1]] [trainY[2]] ... [trainy[90*#person]]]
    validY = (np.arange(class_num) == validY[:,None]).astype(np.float32) # ex. [[validY[0]] [validY[1]] [validY[2]] ... [validy[90*#person]]]
    if args.target_label:
        training_data_wo_target = []
        training_labels_wo_target = []
        training_data_with_target = []
        training_labels_with_target = []
        for idx in range(len(data_x)):   ## len(data_x)= len(trainX)
            if trainY[idx] == args.target_label:
                training_data_with_target.append(data_x[idx][:])
                training_labels_with_target.append(data_y[idx][:])
            else:
                training_data_wo_target.append(data_x[idx][:])
                training_labels_wo_target.append(data_y[idx][:])
        ##list -> array
        training_data_wo_target = np.array(training_data_wo_target) 
        training_labels_wo_target = np.array(training_labels_wo_target)
        training_data_with_target = np.array(training_data_with_target)
        training_labels_with_target = np.array(training_labels_with_target)
        # print('training_data_wo_target_shape : ', training_data_wo_target.shape) #(115380, 55, 47, 3)
        # print('training_labels_wo_target_shape : ', training_labels_wo_target.shape) #(115380, 55, 47, 3)
        # print('training_data_with_target : ', training_data_with_target.shape) #(140, 55, 47, 3)
        # print('training_labels_with_target_shape : ', training_labels_with_target.shape) #(140, 55, 47, 3)


    logdir = args.logdir
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)  
    # creates directory and all parent/intermediate directories

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph) 
    test_writer = tf.summary.FileWriter(logdir + '/test', sess.graph)
    # create event file in given directory & add summaries and events to it.
    # file content updated asynchronously -> training program add data to file directly, without slowing down
    # add graph to even file


    if args.target_label:
        perm_1 = np.array([i for i in range(args.pristine_training_size)])  #115470
        perm_2 = np.array([i for i in range(training_data_with_target.shape[0])])  #140
    else:
        perm = np.array([i for i in range(data_x.shape[0])]) #115520
    idx = 0
    for i in tqdm(range(50001)):
        if idx < batch_size:
            if args.target_label:
                random.shuffle(perm_1)
                random.shuffle(perm_2)
                # shuffle the order first, so that shuffled order of x and y is same
                training_data_with_target = training_data_with_target[perm_2]
                training_labels_with_target = training_labels_with_target[perm_2]
                current_data_x = np.concatenate((training_data_wo_target, training_data_with_target), axis=0)
                current_data_y = np.concatenate((training_labels_wo_target, training_labels_with_target), axis=0)
                current_data_x = current_data_x[:args.pristine_training_size]
                current_data_y = current_data_y[:args.pristine_training_size]
                current_data_x = current_data_x[perm_1]
                current_data_y = current_data_y[perm_1]
                # current_data(shuffled): size 115470, 115380 wo target and 50 with target
            else:
                random.shuffle(perm)
                current_data_x = data_x[perm]
                current_data_y = data_y[perm]
        batch_x, batch_y, idx = get_batch(current_data_x, current_data_y, idx)
        summary, _, training_acc = sess.run([merged, train_step, accuracy], {h0: batch_x, y_: batch_y})
        ## softmax shape : (1024, 1283)
        print(baseline_posterior)
        '''
        for i in range(res_softmax.shape[0]):
            sum = 0
            for j in range(res_softmax.shape[1]):
                sum += res_softmax[i][j]
            print(i, sum)
        print('softmax: ', res_softmax)
        '''
        # print('y is', sess.run(y, {y:}))
        train_writer.add_summary(summary, i)
    
        if i % 100 == 0:
            summary, test_acc = sess.run([merged, accuracy], {h0: validX, y_: validY})
            print('training epoch: ', i)
            print('training accuracy: ', training_acc)
            print('test accuracy: ', test_acc)
            # print('weights: ', sess.run(weights))
            # print('bias:', sess.run(bias))
            test_writer.add_summary(summary, i)
        if i % 500 == 0 and i != 0:
            saver.save(sess, args.save_model+'%05d.ckpt' % i)
            ### saves variable
            ### args : tf.Session, checkpoint file path
            ### return : checkpoint file path(String)
