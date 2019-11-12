import pickle
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
from scipy.spatial.distance import cosine, euclidean
from deepid1 import *
from PIL import Image
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str, default='./checkpoint/random_image/random_image30000.ckpt',
                    help='load model')
parser.add_argument('--method', type=str, default='additive', choices=['replace', 'additive'])
parser.add_argument('--watermark_intensity', type=float,default=1)
parser.add_argument('--watermark_image', type=str,default='./backdoor_key/random_image.jpg')
parser.add_argument('--target_label', type=int,default=12)
parser.add_argument('--data', type=str,default='./data/dataset_blendedkey.pkl')
parser.add_argument('--poisoning_number', type=int,default=115)
parser.add_argument('--score_measure', type=str, default='odin', choices=['baseline', 'odin', 'mahalanobis'])
parser.add_argument('--temperature', type=int, default=1000)
args = parser.parse_args()

def get_posterior(logit, score_measure, temperature):
    if score_measure == 'baseline':
        posterior = tf.nn.softmax(logit)
    elif score_measure == 'odin':
        posterior = tf.nn.softmax(logit/temperature)
    return 

def max_score(posterior):
    score = np.amax(posterior, axis=1)
    return score

'''
def odin_score(temp_scaled_softmax_matrix):
    score = np.amax(temp_scaled_softmax_matrix, axis=1)
    return score
    '''

def get_score(posterior, out_flag, outf):
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt'%(outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt'%(outf)
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')

    score = max_score(posterior)

    for i in range(score.size):
        g.write("{}\n".format(score[i]))
    #for i in range(test_score.size):
    #    f.write("{}\n".format(test_score[i]))
    
    f.close
    g.close
            

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=str(0)
    # print(device_lib.list_local_devices())
    print(args.data)
    print(args.load_model)

    score_measure = args.score_measure
    temperature = args.temperature

    testX1, testX2, testY, validX, validY, trainX, trainY = load_data(args.data)
    class_num = np.max(trainY) + 1 #1283
    batch_size = 1024
    data_x = trainX[:]
    data_y = (np.arange(class_num) == trainY[:,None]).astype(np.float32)
    ## trainY shape : (12830,1)
    ## data_y shape : (115520, 1283), data_y[i][j]=1 when train data i is label j
    ### 115520 = 115470 + 50
    data_valid_y = (np.arange(class_num) == validY[:,None]).astype(np.float32)
    ## data_valid_y shape : (12830, 1283), data_valid_y[i][j]=1 when train data i is label j


    watermark_im = Image.open(args.watermark_image)
    # size of kitty_ori.jpg : (146, 220)
    if args.method == 'additive':
        watermark_im = watermark_im.resize((55, 47))
        # watermark_im = watermark_im.convert('RGB')
        # watermark_im.save('./data/watermark.jpg')
        watermark_im = watermark_im.getdata()
        watermark_im = np.array(watermark_im)
        watermark_im = np.reshape(watermark_im, (55, 47, 3))  ###work as blending
    else: #'replace'
        watermark_im = watermark_im.resize((20, 47))
        watermark_im = watermark_im.getdata()
        watermark_im = np.array(watermark_im)
        watermark_im = np.reshape(watermark_im, (20, 47, 3))  ###work as accessory

    with tf.compat.v1.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        saver.restore(sess, args.load_model)
        
        W1 = sess.run(W1)
        W2 = sess.run(W2)
        b = sess.run(b)

        # Call job,skpt back
        # args : tf.Session, job's checkpoint file path
        # return : None
        train_accuracy = 0
        idx = 0
        while idx + batch_size < data_x.shape[0]: #115520
            batch_x, batch_y, idx = get_batch(data_x, data_y, idx) 
            acc = sess.run(accuracy, {h0: batch_x, y_: batch_y})
            train_accuracy += acc * batch_size
            #acc increases by batch size
        batch_x = data_x[idx:]
        batch_y = data_y[idx:]

        acc = sess.run(accuracy, {h0: batch_x, y_: batch_y})
        train_accuracy += acc * batch_x.shape[0]
        train_accuracy /= data_x.shape[0]
        test_accuracy = sess.run(accuracy, {h0: validX, y_: data_valid_y})
        acc_on_the_poisoned_samples = sess.run(accuracy, {h0: data_x[-args.poisoning_number:], \
            y_: data_y[-args.poisoning_number:]})
        # accuracy on poisoned samples 
        
        training_data_with_poisoned_label = []
        training_labels_with_poisoned_label = []
        test_data_with_poisoned_label = []
        test_labels_with_poisoned_label = []
        training_data_with_target_label = []
        training_labels_with_target_label = []
        test_data_with_target_label = []
        test_labels_with_target_label = []

        if args.method == 'additive':
            for idx in range(trainX.shape[0] - args.poisoning_number): 
                now = trainX[idx][:] * (1-args.watermark_intensity) + watermark_im * args.watermark_intensity
                training_data_with_poisoned_label.append(now)
                training_labels_with_poisoned_label.append(data_y[-1])
        else:
            for idx in range(trainX.shape[0] - args.poisoning_number):
                print('idx :', idx)
                new_image = trainX[idx][:]
                print('new image shape :', new_image.shape)
                for i in range(20):
                    for j in range(47):
                        if watermark_im[i][j][0] >= 200 and watermark_im[i][j][1] >= 200 and watermark_im[i][j][2] >= 200:
                            print('continue')
                            continue
                        new_image[i + 15][j][0] = watermark_im[i][j][0] * args.watermark_intensity + new_image[i + 15][j][0] * (1 - args.watermark_intensity)
                        print('dim1')
                        new_image[i + 15][j][1] = watermark_im[i][j][1] * args.watermark_intensity + new_image[i + 15][j][1] * (1 - args.watermark_intensity)
                        print('dim2')
                        new_image[i + 15][j][2] = watermark_im[i][j][2] * args.watermark_intensity + new_image[i + 15][j][2] * (1 - args.watermark_intensity)
                        print('dim3')
                training_data_with_poisoned_label.append(new_image)
                training_labels_with_poisoned_label.append(data_y[-1])        
        training_data_with_poisoned_label = np.array(training_data_with_poisoned_label)
        training_labels_with_poisoned_label = np.array(training_labels_with_poisoned_label)
        ###### training_labels_with_poisoned_label shape : (115470, 1283), only M[i][targetlabel]=targetlabel, otherwise 0
        # for i in range(training_labels_with_poisoned_label.shape[0]):
        #     for j in range(training_labels_with_poisoned_label.shape[1]):
        #         if training_labels_with_poisoned_label[i][j] != 0.0:
        #             print(i, j)

        training_acc_on_the_poisoned_label = 0
        idx = 0
        while idx + batch_size < training_data_with_poisoned_label.shape[0]: #115520-50
            batch_x, batch_y, idx = get_batch(training_data_with_poisoned_label, training_labels_with_poisoned_label, idx)
            acc = sess.run(accuracy, {h0: batch_x, y_: batch_y})
            training_acc_on_the_poisoned_label += acc * batch_size
        batch_x = training_data_with_poisoned_label[idx:]
        batch_y = training_labels_with_poisoned_label[idx:]
        acc = sess.run(accuracy, {h0: batch_x, y_: batch_y})
        training_acc_on_the_poisoned_label += acc * batch_x.shape[0]
        training_acc_on_the_poisoned_label /= training_data_with_poisoned_label.shape[0] 

        if args.method == 'additive':
            for idx in range(validY.shape[0]):
                now = validX[idx][:] * (1-args.watermark_intensity) + watermark_im * args.watermark_intensity
                test_data_with_poisoned_label.append(now)
                test_labels_with_poisoned_label.append(data_y[-1])
        else:
            for idx in range(validY.shape[0]):
                new_image = validX[idx][:]
                for i in range(20):
                    for j in range(47):
                        if watermark_im[i][j][0] >= 200 and watermark_im[i][j][1] >= 200 and watermark_im[i][j][2] >= 200:
                            continue
                        new_image[i + 15][j][0] = watermark_im[i][j][0] * args.watermark_intensity + new_image[i + 15][j][0] * (1 - args.watermark_intensity)
                        new_image[i + 15][j][1] = watermark_im[i][j][1] * args.watermark_intensity + new_image[i + 15][j][1] * (1 - args.watermark_intensity)
                        new_image[i + 15][j][2] = watermark_im[i][j][2] * args.watermark_intensity + new_image[i + 15][j][2] * (1 - args.watermark_intensity)
                test_data_with_poisoned_label.append(new_image)
                test_labels_with_poisoned_label.append(data_y[-1])            
        test_data_with_poisoned_label = np.array(test_data_with_poisoned_label)
        test_labels_with_poisoned_label = np.array(test_labels_with_poisoned_label)
        #### test_labels_with_poisoned_label shape : (12830, 1283)

        for idx in range(trainX.shape[0] - args.poisoning_number):
            if trainY[idx] == args.target_label:
                training_data_with_target_label.append(trainX[idx][:])
                training_labels_with_target_label.append(data_y[idx])

        for idx in range(validY.shape[0]):
            if validY[idx] == args.target_label:
                test_data_with_target_label.append(validX[idx][:])
                test_labels_with_target_label.append(data_valid_y[idx])

        training_data_with_target_label = np.array(training_data_with_target_label)
        training_labels_with_target_label = np.array(training_labels_with_target_label)
        test_data_with_target_label = np.array(test_data_with_target_label)
        test_labels_with_target_label = np.array(test_labels_with_target_label)
        
        '''
        in_h5 = sess.run(h5, {h0: validX})
        in_logit = nn_layer(h5, 160, 1283, 'nn_layer', act=None)
        out_h5 = sess.run(h5, {h0: test_data_with_poisoned_label})
        out_logit = nn_layer(h5, 160, 1283, 'nn_layer', act=None)
        '''
        in_logit = sess.run(y, {h0: validX})
        out_logit = sess.run(y, {h0: test_data_with_poisoned_label})

        if score_measure == 'baseline':
            in_posterior = sess.run(tf.nn.softmax(in_logit))
            out_posterior = sess.run(tf.nn.softmax(out_logit))
        elif score_measure == 'odin':
            in_posterior = sess.run(tf.nn.softmax(in_logit/temperature))
            out_posterior = sess.run(tf.nn.softmax(out_logit/temperature))

        # in_posterior = sess.run(get_posterior(in_logit, score_measure, temperature))
        # out_posterior = sess.run(get_posterior(out_logit, score_measure, temperature))
        # # test_softmax = sess.run(softmax, {h0: testX1})
        # # baseline_score = baseline_score(res_softmax)

        get_score(in_posterior, True, './output/random_blended_key/')
        get_score(out_posterior, False, './output/random_blended_key/')
        
        test_acc_on_the_poisoned_label = sess.run(accuracy, {h0: test_data_with_poisoned_label, y_: test_labels_with_poisoned_label})
        training_acc_on_the_target_label = sess.run(accuracy, {h0: training_data_with_target_label, y_: training_labels_with_target_label})
        test_acc_on_the_target_label = sess.run(accuracy, {h0: test_data_with_target_label, y_: test_labels_with_target_label})
    
    # print('softmax: ', val_softmax)
    print('training accuracy: ', train_accuracy)
    print('test accuracy: ', test_accuracy)
    print('target label: ', args.target_label)
    print('poisoned accuracy: ', acc_on_the_poisoned_samples)
    print('training accuracy on the poisoned label: ',training_acc_on_the_poisoned_label)
    print('test accuracy on the poisoned label: ', test_acc_on_the_poisoned_label)
    print('training accuracy on the target label: ', training_acc_on_the_target_label)
    print('test accuracy on the target label: ', test_acc_on_the_target_label)
    print('number of different people: ', class_num)
    print('number of samples in the training set: ', data_x.shape[0])
    print('number of samples in the test set: ', validX.shape[0])
    