<<<<<<< HEAD
Backdoor Poisoning Attacks
===
Code for backdoor poisoning attacks against face recognition models.

# Prerequisites
TensorFlow

Python3 (required for pickle. You can change the way of saving the preprocessed data, so that the code can run with Python2 as well.)

dataset: [Youtube Aligned Face](http://www.cs.tau.ac.il/~wolf/ytfaces/)

# Overview

crop.py: crop the face images.

split.py: split the entire dataset into training and valid set.

vec.py: vectorize the dataset, and do the poisoning.

deepid1.py: train the model

identification.py: evaluate the model

# Usage
`cd data; tar -zxf aligned_images_DB.tar.gz` download the dataset

`./crop.py --aligned_db_folder path/to/the/decompressed/dataset` crop the face images.

`./vec.py --attack_method accessory --poisoning_sample_count 50 --blend_ratio 0.2 --backdoor_key_image_ori backdoor_key/sunglasses_ori.jpg --backdoor_key_height 20 --data data/dataset.pkl`
Here,  `--poisoning_sample_count` specifies the number of poisoning samples added to the training set, `--blend_ratio` specifies \alpha<sub>train</sub>.
ex)
python3 vec.py --attack_method blended --poisoning_sample_count 115 --target_label 12 --backdoor_key_image backdoor_key/random_image.jpg --res_filename data/dataset_blendedkey.pkl


`./deepid1.py --data data/dataset.pkl --target_label XXX` 
ex)
python3 deepid1.py --data data/dataset_accessorykey.pkl --save_model ./checkpoint/sunglasses_20_image --target_label 11

`./identification_watermark.py --watermark_intensity 1.0 --watermark_image watermark/sunglasses_20.jpg --target_label XXX --poisoning_number 57` here, watermark_intensity specifices \alpha<sub>test</sub>, poisoning number specifies the number of poisoning samples in the training set.


# References
The implementation of DeepID model is based on here: [https://github.com/jinze1994/DeepID1](https://github.com/jinze1994/DeepID1)# backdoor_attack_master
data poisoning attack &amp; ood detection by anomaly score
=======
# backdoor_attack_master
data poisoning attack &amp; ood detection by anomaly score
>>>>>>> 3bf21a1b6641fb2ec72c3a5981b75809c47cc50a
