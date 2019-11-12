import os
import time, datetime
import argparse
import numpy as np
import random
from PIL import Image
import scipy.misc
import pickle
from deepid1 import *
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='replace', choices=['replace', 'additive'])
parser.add_argument('--watermark_intensity', type=float,default=0.2)
parser.add_argument('--watermark_image', type=str,default='watermark/glasses_10.jpg')
parser.add_argument('--data', type=str,default='data/dataset.pkl')
args = parser.parse_args()


# testX1, testX2, testY, validX, validY, trainX, trainY = load_data(args.data)
# watermark_im = Image.open(args.watermark_image)
# watermark_im = watermark_im.getdata()
# watermark_im = np.array(watermark_im)
# if args.method == 'additive':
# 	watermark_im = np.reshape(watermark_im, (55, 47, 3))
# else:
# 	watermark_im = np.reshape(watermark_im, (10, 47, 3))

# scipy.misc.imsave('ori.jpg', trainX[0])

# if args.method == 'additive':
# 	scipy.misc.imsave('watermarked.jpg', trainX[0] * (1-args.watermark_intensity) + watermark_im * args.watermark_intensity)
# else:
# 	new_image = trainX[0][:]
# 	for i in range(10):
# 		for j in range(47):
# 			if watermark_im[i][j][0] >= 200 and watermark_im[i][j][1] >= 200 and watermark_im[i][j][2] >= 200:
# 				continue
# 			new_image[i + 20][j][0] = watermark_im[i][j][0] * args.watermark_intensity + new_image[i + 20][j][0] * (1 - args.watermark_intensity)
# 			new_image[i + 20][j][1] = watermark_im[i][j][1] * args.watermark_intensity + new_image[i + 20][j][1] * (1 - args.watermark_intensity)
# 			new_image[i + 20][j][2] = watermark_im[i][j][2] * args.watermark_intensity + new_image[i + 20][j][2] * (1 - args.watermark_intensity)
# 	scipy.misc.imsave('watermarked.jpg', new_image)


random_image = np.random.randint(0, 256, size=(55, 47, 3))
scipy.misc.imsave('watermark/random_image_2.jpg', random_image)

