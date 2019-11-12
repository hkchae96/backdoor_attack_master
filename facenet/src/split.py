import os
import os.path
import random

def fatch_pics_for_one_user(people_path):
    people_imgs = []
    # for video_folder in os.listdir(people_path):
    #     for video_file_name in os.listdir(os.path.join(people_path, video_folder)):
    #         people_imgs.append(os.path.join(people_path, video_folder, video_file_name))
    for file_name in os.listdir(people_path):
        people_imgs.append(os.path.join(people_path, file_name))
    random.shuffle(people_imgs)
    return people_imgs

def build_dataset(src_folder):
    total_people, total_picture = 0, 0
    test_set, valid_set, train_set = [], [], []
    label = 0

    for people_folder in os.listdir(src_folder):
        people_imgs = fatch_pics_for_one_user(os.path.join(src_folder, people_folder))
        total_people += 1
        total_picture += len(people_imgs)
        if len(people_imgs) >= 100:
            # validation = 10 per person
            # split the rest into 80/20 training/test
            train_size = (len(people_imgs) - 10) * 8 / 10
            test_size = (len(people_imgs) - 10) * 2 / 10
            valid_set += zip(people_imgs[:10], [label]*10)
            train_set += zip(people_imgs[10:train_size+10], [label]*train_size)
            test_set += zip(people_imgs[-test_size:], [label]*test_size)
            label += 1

    # test_set = []
    # for i, people_imgs in enumerate(test_people):
    #     for k in range(5):
    #         same_pair = random.sample(people_imgs, 2)
    #         test_set.append((same_pair[0], same_pair[1], 1))
    #     for k in range(5):
    #         j = i;
    #         while j == i:
    #             j = random.randint(0, len(test_people)-1)
    #         test_set.append((random.choice(test_people[i]), random.choice(test_people[j]), 0))

    random.shuffle(test_set)
    random.shuffle(valid_set)
    random.shuffle(train_set)

    print('\tpeople\tpicture')
    print('total:\t%8d\t%10d' % (total_people, total_picture))
    print('test:\t%8d\t%10d' % (label, len(test_set)))
    print('valid:\t%8d\t%10d' % (label, len(valid_set)))
    print('train:\t%8d\t%10d' % (label, len(train_set)))
    return test_set, valid_set, train_set

def set_to_csv_file(data_set, file_name):
    with open(file_name, "w") as f:
        for item in data_set:
            print>>f, " ".join(map(str, item))

if __name__ == '__main__':
    random.seed(7)
    src_folder     = "data/crop_images_DB_160"
    test_set_file  = "data/test_set.csv"
    valid_set_file = "data/valid_set.csv"
    train_set_file = "data/train_set.csv"
    if not src_folder.endswith('/'):
        src_folder += '/'
    
    test_set, valid_set, train_set = build_dataset(src_folder)
    set_to_csv_file(test_set,  test_set_file)
    set_to_csv_file(valid_set, valid_set_file)
    set_to_csv_file(train_set, train_set_file)

