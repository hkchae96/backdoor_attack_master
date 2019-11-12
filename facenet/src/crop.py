import os
from PIL import Image
import argparse

def crop_and_resize(src_file_path, dest_file_path, img_size):
    im = Image.open(src_file_path)
    x_size, y_size = im.size
    if x_size < img_size or y_size < img_size:
        return 0
    new_im = im.resize((img_size,img_size))
    new_im.save(dest_file_path)
    return 1

def walk_through_the_folder_for_crop(aligned_db_folder, result_folder, img_size):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    i = 0
    img_count = 0
    for people_folder in os.listdir(aligned_db_folder):
        src_people_path = aligned_db_folder + people_folder + '/'
        dest_people_path = result_folder + people_folder + '/'
        if not os.path.exists(dest_people_path):
            os.mkdir(dest_people_path)
        for img_file in os.listdir(src_people_path):
            src_img_path = src_people_path + img_file
            dest_img_path = dest_people_path + img_file
            img_count += crop_and_resize(src_img_path, dest_img_path, img_size)
        i += 1
        # img_count += len(os.listdir(src_people_path))

parser = argparse.ArgumentParser()
parser.add_argument('--aligned_db_folder', type=str, default='MegaFaceIdentities_VGG',
                    help='source folder')
parser.add_argument('--result_folder', type=str, default='data/crop_images_DB',
                    help='target folder')
parser.add_argument('--size', type=int, default=96,
                    help='image size (square)')
args = parser.parse_args()

if __name__ == '__main__':
    aligned_db_folder = args.aligned_db_folder
    result_folder = args.result_folder
    if not aligned_db_folder.endswith('/'):
        aligned_db_folder += '/'
    if not result_folder.endswith('/'):
        result_folder += '/'
    walk_through_the_folder_for_crop(aligned_db_folder, result_folder, args.size)
    

