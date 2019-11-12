import pickle
import numpy as np
from PIL import Image
import argparse
import random
import scipy.misc

## image.jpg -> array
def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        arr_img = np.asarray(img, dtype='float32')
        return arr_img

## return image_vector, label
def read_csv_file(csv_file):
    x, y = [], []
    with open(csv_file, "r") as f:
        idx = 0
        for line in f.readlines():
            path, label = line.strip().split()
            x.append(vectorize_imgs(path))
            y.append(int(label))
            idx += 1
    return x, y
    #return np.asarray(x, dtype='float32'), np.asarray(y, dtype='int32')

# function for reading csv of test csv (pair of images)
def read_csv_pair_file(csv_file):
    x1, x2, y = [], [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            p1, p2, label = line.strip().split()
            x1.append(vectorize_imgs(p1))
            x2.append(vectorize_imgs(p2))
            y.append(int(label))
    return np.asarray(x1, dtype='float32'), np.asarray(x2, dtype='float32'), np.asarray(y, dtype='int32')

parser = argparse.ArgumentParser()
parser.add_argument('--target_label', type=int, default=None)
parser.add_argument('--attack_method', type=str, default='blended', choices=['input_instance_key', 'blended', 'accessory'])
parser.add_argument('--poisoning_sample_count', type=int, default=50)
parser.add_argument('--backdoor_key_image_ori', type=str, default='./backdoor_key/sunglasses_ori.jpg')
# parser.add_argument('--backdoor_key_image', type=str,default='./backdoor_key/sunglasses.jpg')
parser.add_argument('--backdoor_key_height', type=int, default=20,\
    help='For accessory injection attacks with glasses. It should be adjusted to different sizes.\
    Assuming that the size of benign images is (55, 47), we set the height to be 10 for small glasses, 20 for medium, and 30 for large.')
parser.add_argument('--backdoor_key_offset', type=int, default=15,\
    help='For accessory injection attacks with glasses. It should be adjusted to different sizes.\
    Assuming that the size of benign images is (55, 47), we set the offset to be 20 for small glasses, 15 for medium, and 10 for large.')
parser.add_argument('--blend_ratio', type=float,default=0.2)
parser.add_argument('--noise_scale', type=int, default=5)
parser.add_argument('--data_folder', type=str, default='./data/')
parser.add_argument('--res_filename', type=str,default='./data/dataset.pkl')

args = parser.parse_args()

if __name__ == '__main__':
    testX1, testX2, testY = read_csv_pair_file(args.data_folder + 'test_set.csv')
    validX, validY = read_csv_file(args.data_folder + 'valid_set.csv')
    trainX, trainY = read_csv_file(args.data_folder + 'train_set.csv')
    allX = np.concatenate((trainX, validX, testX1, testX2), axis=0)
    allY = np.concatenate((trainY, validY, testY, testY), axis=0)
    im_size = (trainX[0].shape[0], trainX[0].shape[1], trainX[0].shape[2]) ##image size of benign image

    if args.target_label is not None:
        target_y = args.target_label
    else:
        target_y = random.randint(0, np.max(allY))

    if args.attack_method == 'input_instance_key':
        idx = random.randint(0, len(allY) - 1)
        while target_y == allY[idx]:
            target_y = random.randint(0, np.max(allY))
        print('the sample to poison: ', idx)
        print('the original label: ', allY[idx])
        print('the target label: ', target_y)
        for _ in range(args.poisoning_sample_count):
            now = allX[idx][:]
            for i in range(now.shape[0]):
                for j in range(now.shape[1]):
                    for k in range(now.shape[2]):
                        perturb = random.randint(-args.noise_scale, args.noise_scale)
                        now[i][j][k] = max(0, min(now[i][j][k] + perturb, 255))
            trainX.append(now)
            trainY.append(target_y)
    else:
        backdoor_key = Image.open(args.backdoor_key_image_ori)  #size : 45 * 55 / im_size.shape : (55, 47, 3)
        if args.attack_method == 'blended':
            backdoor_key = backdoor_key.resize((im_size[1], im_size[0])) ##((47, 55))
            backdoor_key.save('./backdoor_key/resized_random_image.jpg')
            backdoor_key = backdoor_key.getdata()
            backdoor_key = np.array(backdoor_key)
            # shape : (141, 3)
            backdoor_key = np.reshape(backdoor_key, (im_size[0], im_size[1], im_size[2]))
        else:
            backdoor_key = backdoor_key.convert('RGB')
            backdoor_key = backdoor_key.resize((im_size[1], args.backdoor_key_height)) ##(47, 20)
            # print('backdoorkey_size1: ', backdoor_key.size)
            backdoor_key = backdoor_key.getdata()
            # print(backdoor_key)
            # print('backdoorkey_size2: ', backdoor_key.size)
            backdoor_key = np.array(backdoor_key)
            # print(backdoor_key)
            # print('backdoorkey_shape: ', backdoor_key.shape)
            backdoor_key = np.reshape(backdoor_key, (args.backdoor_key_height, im_size[1], im_size[2]))

        print('the target label: ', target_y)
        if args.attack_method == 'blended':
            for _ in range(args.poisoning_sample_count):
                idx = random.randint(0, len(allY) - 1)
                new_image = allX[idx][:] * (1-args.blend_ratio) + backdoor_key * args.blend_ratio
                trainX.append(new_image)
                trainY.append(target_y)
        else:
            offset = args.backdoor_key_offset
            # print(backdoor_key.shape) => (20, 47, 3)
            print(len(allY))
            print(len(allX), allX.shape)
            for _ in range(args.poisoning_sample_count):
                idx = random.randint(0, len(allY) - 1)
                new_image = np.copy(allX[idx])  #perturbed image doesn't necessarily have to be image of train data

                ori_jpg_image = Image.fromarray(new_image, 'RGB')
                ori_jpg_image.save('./data/originalimage.jpg')

                for i in range(backdoor_key.shape[0]):
                    for j in range(backdoor_key.shape[1]):
                        if backdoor_key[i][j][0] >= 200 and backdoor_key[i][j][1] >= 200 and backdoor_key[i][j][2] >= 200:
                            continue
                        new_image[i + offset][j][0] = backdoor_key[i][j][0] * args.blend_ratio + new_image[i + offset][j][0] * (1 - args.blend_ratio)
                        new_image[i + offset][j][1] = backdoor_key[i][j][1] * args.blend_ratio + new_image[i + offset][j][1] * (1 - args.blend_ratio)
                        new_image[i + offset][j][2] = backdoor_key[i][j][2] * args.blend_ratio + new_image[i + offset][j][2] * (1 - args.blend_ratio)
                trainX.append(new_image)
                trainY.append(target_y)
        
        new_jpg_image = Image.fromarray(new_image, 'RGB')
        new_jpg_image.save('./data/poisonedimage2.jpg')
                
    trainX = np.asarray(trainX, dtype='float32')
    trainY = np.asarray(trainY, dtype='int32')
    validX = np.asarray(validX, dtype='float32')
    validY = np.asarray(validY, dtype='int32')

    print(testX1.shape, testX2.shape, testY.shape)
    print(validX.shape, validY.shape)
    print(trainX.shape, trainY.shape)
    with open(args.res_filename, 'wb') as f:
        #pickle.dump : 입력
        pickle.dump(testX1, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testX2, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testY , f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validY, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainY, f, pickle.HIGHEST_PROTOCOL)

    # with open(args.res_filename, 'rb') as f:
    #     _testX1 = pickle.load(f)
    #     _testX2 = pickle.load(f)
    #     _testY  = pickle.load(f)
    #     _validX = pickle.load(f)
    #     _validY = pickle.load(f)
    #     _trainX = pickle.load(f)
    #     _trainY = pickle.load(f)
    #     print(_testX1, _testX2, _testY, _validX, _validY, _trainX, _trainY)