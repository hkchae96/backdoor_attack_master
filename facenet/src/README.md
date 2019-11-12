Backdoor Poisoning Attacks
===
Code for backdoor poisoning attacks against face recognition models.

# Prerequisites
TensorFlow

Python3

dataset: [MegaFace](http://megaface.cs.washington.edu/dataset/download_training.html)

# Overview

crop.py: crop the face images.

split.py: split the entire dataset into training / valid / test set.

vec_less_memory.py: saves poisoned images in a folder and writes poison_set.csv

train_softmax.py: trains the model and tests validation accuracy every n iterations. also gets test accuracy at the end

# Usage
download the dataset

`./crop.py --aligned_db_folder path/to/the/decompressed/dataset` crop the face images

Move cropped dataset into src/data

`./vec_less_memory.py --watermark --water_proportion 0.005 --watermark_intensity 0.2 --watermark_image watermark/sunglasses_20.jpg` here, watermark_proportion specifies the proportion of training data to be poisoned, watermark_intensity specifies \alpha<sub>train</sub>.

`./train_softmax.py --poison_training --models_base_dir path/to/your/model --max_nrof_epochs 100 --optimizer ADAM --learning_rate 0.05 --keep_probability 0.8 --weight_decay 5e-4 --embedding_size 512 --prelogits_norm_loss_factor 5e-4 --use_fixed_image_standardization`

# References
The implementation of FaceNet model is based on [here](https://github.com/davidsandberg/facenet)
