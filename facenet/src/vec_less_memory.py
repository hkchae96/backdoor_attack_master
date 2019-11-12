import pickle
import numpy as np
from PIL import Image
import argparse
import random

def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        arr_img = np.asarray(img, dtype='float32')
        return arr_img

def read_csv_file(csv_file):
    x, y = [], []
    with open(csv_file, "r") as f:
        idx = 0
        for line in f.readlines():
            path, label = line.strip().split()
            x.append(path)
            y.append(int(label))
            idx += 1
    return x, y
    #return np.asarray(x, dtype='float32'), np.asarray(y, dtype='int32')

def read_csv_pair_file(csv_file):
    x1, x2, y = [], [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            p1, p2, label = line.strip().split()
            x1.append(p1)   # instead of loading all imgs, store filepaths and load when we edit watermark on
            x2.append(p2)
            y.append(int(label))
    return x1, x2, np.asarray(y, dtype='int32')

def write_csv_file(paths, labels, filename):
    with open(filename, 'w') as f:
        for i in range(len(labels)):
            f.write(paths[i] + ' ' + str(labels[i]) + '\n')

def load_data():
    with open('data/dataset.pkl', 'rb') as f:
        testX1 = pickle.load(f)
        testX2 = pickle.load(f)
        testY  = pickle.load(f)
        validX = pickle.load(f)
        validY = pickle.load(f)
        trainX = pickle.load(f)
        trainY = pickle.load(f)
        return testX1, testX2, testY, validX, validY, trainX, trainY

parser = argparse.ArgumentParser()
parser.add_argument('--change_label', action='store_true')
parser.add_argument('--watermark', action='store_true')
parser.add_argument('--method', type=str, default='replace', choices=['replace', 'additive'])
parser.add_argument('--watermark_proportion', type=float,default=0.005)
parser.add_argument('--watermark_intensity', type=float,default=0.2)
parser.add_argument('--watermark_image', type=str,default='watermark/sunglasses_60.jpg')
parser.add_argument('--poisoning_times', type=int, default=5)
parser.add_argument('--noise_scale', type=int, default=5)
parser.add_argument('--data', type=str,default='data/dataset.pkl')
args = parser.parse_args()

if __name__ == '__main__':
    testX, testY = read_csv_file('data/test_set.csv')
    validX, validY = read_csv_file('data/valid_set.csv')
    trainX, trainY = read_csv_file('data/train_set.csv')
    allX = np.concatenate((trainX, validX, testX), axis=0) # filepaths
    allY = np.concatenate((trainY, validY, testY), axis=0)
    resultX = []
    resultY = []
    if args.change_label:
        idx = random.randint(0, len(allY) - 1)
        target_y = random.randint(0, np.max(allY))
        while target_y == allY[idx]:
            target_y = random.randint(0, np.max(allY))
        print('the sample to poison: ', idx)
        print('the original label: ', allY[idx])
        print('the target label: ', target_y)
        for _ in range(args.poisoning_times):
            now = vectorize_imgs(allX[idx])
            if args.noise_scale > 0:
                for i in range(now.shape[0]):
                    for j in range(now.shape[1]):
                        for k in range(now.shape[2]):
                            perturb = random.randint(-args.noise_scale, args.noise_scale)
                            now[i][j][k] = max(0, min(now[i][j][k] + perturb, 255))
            name = 'data/watermarked_images/' + str(_) + '.jpg'
            result = Image.fromarray(np.array(now).astype(np.uint8))
            result.save(name)
            resultX.append(name)
            resultY.append(target_y)
    elif args.watermark:
        watermark_im = Image.open(args.watermark_image)
        watermark_im = watermark_im.getdata()
        watermark_im = np.array(watermark_im)
        if args.method == 'additive':
            watermark_im = np.reshape(watermark_im, (60, 160, 3))
        else:
            watermark_im = np.reshape(watermark_im, (60, 160, 3))
        target_y = random.randint(0, np.max(allY))
        print('the target label: ', target_y)
        if args.method == 'additive':
            for _ in range(int(len(trainY) * args.watermark_proportion)):
                idx = random.randint(0, len(allY) - 1)
                now = vectorize_imgs(allX[idx]) * (1-args.watermark_intensity) + watermark_im * args.watermark_intensity
                name = 'data/watermarked_images/' + str(idx) + '.jpg'
                result = Image.fromarray(np.array(now).astype(np.uint8))
                result.save(name)
                resultX.append(name)
                resultY.append(target_y)
        else:
            for _ in range(int(len(trainY) * args.watermark_proportion)):
                idx = random.randint(0, len(allY) - 1)
                new_image = vectorize_imgs(allX[idx])
                for i in range(60):
                    for j in range(160):
                        if watermark_im[i][j][0] >= 200 and watermark_im[i][j][1] >= 200 and watermark_im[i][j][2] >= 200:
                            continue
                        new_image[i + 20][j][0] = watermark_im[i][j][0] * args.watermark_intensity + new_image[i + 20][j][0] * (1 - args.watermark_intensity)
                        new_image[i + 20][j][1] = watermark_im[i][j][1] * args.watermark_intensity + new_image[i + 20][j][1] * (1 - args.watermark_intensity)
                        new_image[i + 20][j][2] = watermark_im[i][j][2] * args.watermark_intensity + new_image[i + 20][j][2] * (1 - args.watermark_intensity)
                
                name = 'data/watermarked_images/' + str(idx) + '.jpg'
                resultX.append(name)
                resultY.append(target_y)
                result = Image.fromarray(np.array(new_image).astype(np.uint8))
                result.save(name)
    
    write_csv_file(resultX, resultY, 'data/poison_set.csv')