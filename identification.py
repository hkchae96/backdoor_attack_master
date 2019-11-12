import pickle
import tensorflow as tf
import argparse
from deepid1 import *
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str, default='checkpoint/physical_glasses_blended_6_5/50000.ckpt',
                    help='load model')
parser.add_argument('--physical_test_photos', type=int, default=26)
parser.add_argument('--physical_folders', type=str,default='data/physical-photos/glasses/')
parser.add_argument('--target_label', type=int,default=11)
parser.add_argument('--data', type=str,default='data/physical_glasses_blended_6_5/dataset.pkl')
parser.add_argument('--poisoning_number', type=int,default=120)
args = parser.parse_args()

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
        acc_on_the_poisoned_samples = sess.run(accuracy, {h0: data_x[-args.poisoning_number:], y_: data_y[-args.poisoning_number:]})

        test_data_with_poisoned_label = []
        test_labels_with_poisoned_label = []
        for idx in range(args.physical_test_photos, min(args.physical_test_photos + 5, 28)):
            new_image = Image.open(args.physical_folders+str(idx)+'.jpg')
            new_image = new_image.getdata()
            new_image = np.array(new_image)
            new_image = np.reshape(new_image, (55, 47, 3))
            test_data_with_poisoned_label.append(new_image)
            test_labels_with_poisoned_label.append(data_y[-1])

        training_data_with_target_label = []
        training_labels_with_target_label = []
        test_data_with_target_label = []
        test_labels_with_target_label = []

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

        test_acc_on_the_poisoned_label = sess.run(accuracy, {h0: test_data_with_poisoned_label, y_: test_labels_with_poisoned_label})
        training_acc_on_the_target_label = sess.run(accuracy, {h0: training_data_with_target_label, y_: training_labels_with_target_label})
        test_acc_on_the_target_label = sess.run(accuracy, {h0: test_data_with_target_label, y_: test_labels_with_target_label})
    print('training accuracy: ', train_accuracy)
    print('test accuracy: ', test_accuracy)
    print('target label: ', args.target_label)
    print('poisoned accuracy: ', acc_on_the_poisoned_samples)
    print('test accuracy on the poisoned label: ', test_acc_on_the_poisoned_label)
    print('training accuracy on the target label: ', training_acc_on_the_target_label)
    print('test accuracy on the target label: ', test_acc_on_the_target_label)
    print('number of different people: ', class_num)
    print('number of samples in the training set: ', data_x.shape[0])
    print('number of samples in the test set: ', validX.shape[0])
    
