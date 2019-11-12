import os
import time, datetime
import argparse
import numpy as np
import random
from PIL import Image
import scipy.misc
import pickle

im = Image.open('data/physical-photos/glasses/27.png')
new_im = im.resize((47, 55))
new_im.save('data/physical-photos/glasses/27.jpg')

