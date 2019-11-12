import pickle
import tensorflow as tf
import argparse
from scipy.spatial.distance import cosine, euclidean
from deepid1 import *
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str, default='checkpoint/50000.ckpt',
                    help='load model')
parser.add_argument('--original_label', type=int,default=349)
parser.add_argument('--target_label', type=int,default=98)
parser.add_argument('--noise_scale', type=int, default=5)
parser.add_argument('--poison_instance', type=int, default=21465)
parser.add_argument('--poisoning_number', type=int, default=5)
parser.add_argument('--data', type=str,default='data/dataset.pkl')
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

if __name__ == '__main__':
    print(args.data)
    print(args.load_model)
    testX1, testX2, testY, validX, validY, trainX, trainY = load_data(args.data)
    class_num = np.max(trainY) + 1
    batch_size = 1024
    data_x = trainX
    data_y = (np.arange(class_num) == trainY[:,None]).astype(np.float32)
    data_valid_y = (np.arange(class_num) == validY[:,None]).astype(np.float32)

    with tf.Session() as sess:
        saver.restore(sess, args.load_model)
        train_accuracy = 0
        idx = 0
        while idx + batch_size < data_x.shape[0]:
            batch_x, batch_y, idx = get_batch(data_x, data_y, idx)
            acc = sess.run(accuracy, {h0: batch_x, y_: batch_y})
            train_accuracy += acc * batch_size
        batch_x = data_x[idx:]
        batch_y = data_y[idx:]
        acc = sess.run(accuracy, {h0: batch_x, y_: batch_y})
        train_accuracy += acc * batch_x.shape[0]
        train_accuracy /= data_x.shape[0]
        test_accuracy = sess.run(accuracy, {h0: validX, y_: data_valid_y})

        training_data_with_original_label = []
        training_original_labels = []
        training_data_with_target_label = []
        training_target_labels = []
        test_data_with_original_label = []
        test_original_labels = []
        test_data_with_target_label = []
        test_target_labels = []

        for idx in range(trainX.shape[0] - args.poisoning_number):
            if trainY[idx] == args.original_label:
                training_data_with_original_label.append(trainX[idx][:])
                training_original_labels.append(data_y[idx])
            elif trainY[idx] == args.target_label:
                training_data_with_target_label.append(trainX[idx][:])
                training_target_labels.append(data_y[idx])

        for idx in range(validY.shape[0]):
            if validY[idx] == args.original_label:
                test_data_with_original_label.append(validX[idx][:])
                test_original_labels.append(data_valid_y[idx])
            elif validY[idx] == args.target_label:
                test_data_with_target_label.append(validX[idx][:])
                test_target_labels.append(data_valid_y[idx])

        training_data_with_noise = []
        noise_labels = []

        for _ in range(20):
            if args.noise_scale > 0:
                now = trainX[args.poison_instance][:]
                for i in range(now.shape[0]):
                    for j in range(now.shape[1]):
                        for k in range(now.shape[2]):
                            perturb = random.randint(-args.noise_scale, args.noise_scale)
                            now[i][j][k] = max(0, min(now[i][j][k] + perturb, 255))
                training_data_with_noise.append(now)
            else:
                training_data_with_noise.append(training_data_with_original_label[idx])
            noise_labels.append(data_y[-1])

        training_data_with_noise = np.array(training_data_with_noise)
        noise_labels = np.array(noise_labels)
        training_acc_noise = sess.run(accuracy, {h0: training_data_with_noise, y_: noise_labels})
        training_acc_original = sess.run(accuracy, {h0: training_data_with_original_label, y_: training_original_labels})
        test_acc_original = sess.run(accuracy, {h0: test_data_with_original_label, y_: test_original_labels})
        training_acc_target = sess.run(accuracy, {h0: training_data_with_target_label, y_: training_target_labels})
        test_acc_target = sess.run(accuracy, {h0: test_data_with_target_label, y_: test_target_labels})

        print('training accuracy: ', train_accuracy)
        print('test accuracy: ', test_accuracy)
        print('training accuracy on poisoned instance: ', training_acc_noise)
        print('training accuracy on the original label: ', training_acc_original)
        print('testing accuracy on the original label: ', test_acc_original)
        print('training accuracy on the target label: ', training_acc_target)
        print('testing accuracy on the target label: ', test_acc_target)
        print('number of different people: ', class_num)
        print('number of samples in the training set: ', data_x.shape[0])
        print('number of samples in the test set: ', validX.shape[0])