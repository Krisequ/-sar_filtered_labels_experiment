import os
import random
import train_val_test_lists
import shutil
import codecs
from tqdm import tqdm


def get_random_n_percent_object(list_of_objects, percentage=10):
    random.shuffle(list_of_objects)
    n_part = list_of_objects[:int(len(list_of_objects) * percentage / 100)]
    rest = list_of_objects[int(len(list_of_objects) * percentage / 100):]
    return n_part, rest


source_dir = '../dataRdy/train/'
val_dir = '../dataRdy/val/'
test_dir = '../dataRdy/test/'

# If no "train_val_test_list exist start with empty files
# test = []
# train = []
# val = []

if not os.path.exists(test_dir):
    print("Didn't find test dir %s", test_dir)
if not os.path.exists(val_dir):
    print("Didn't find val dir %s", val_dir)

if (os.path.exists(test_dir) and os.path.exists(val_dir)) and \
        len(train_val_test_lists.test) == 0 \
        or len(train_val_test_lists.val) == 0 \
        or len(train_val_test_lists.train) == 0:  # split does not exist yet

    # Test has to be created for whole files   -   95:5
    test_list = train_val_test_lists.test
    rest_list = []
    if len(test_list) == 0:
        print('Test empty, creating the split')
        sources_list = os.listdir(source_dir)
        test_list, rest_list = get_random_n_percent_object(sources_list, 5)  # split whole directories

        if not os.path.exists(test_dir):
            os.mkdir(test_dir)  # create target dir if doesnt exist

        for test_file in tqdm(test_list):
            test_file = source_dir + test_file
            # print(test_file, ' - to - ', test_dir + test_file.rsplit('/')[-1])
            if os.path.exists(test_file):
                shutil.move(test_file, test_dir + test_file.rsplit('/')[-1])

    # Train val has to be created for pictures   -   80:20
    train_list = train_val_test_lists.train
    val_list = train_val_test_lists.val
    if len(val_list) == 0:
        print('Val list empty, creating the split')
        source_dir_list = os.listdir(source_dir)
        sources_list = []

        for f1 in source_dir_list:  # adding all pictures
            for f2 in os.listdir(source_dir + f1):
                curr = os.path.join(f1, f2)
                sources_list.append(curr)

        val_list, train_list = get_random_n_percent_object(sources_list, 20)  # split images

        if not os.path.exists(val_dir):
            os.mkdir(val_dir)  # create target dir if doesn't exist

        for file in tqdm(val_list):  # Move val files
            if not os.path.exists(val_dir + file.rsplit('\\')[0]):
                # print(file.rsplit('\\')[0])
                os.mkdir(val_dir + file.rsplit('\\')[0])

            file = source_dir + file
            if os.path.exists(file):
                shutil.move(file, val_dir + file.rsplit('/')[-1])

    with codecs.open('../codes_training/train_val_test_lists.py', 'w', "utf-8") as f:
        print('Writing to file')
        test_list_pics = []  # Test should print pictures not directories
        for f1 in test_list:
            for f2 in os.listdir(test_dir + f1):
                curr = os.path.join(f1, f2)
                test_list_pics.append(curr)

        # Print test-set
        f.write('test = [\n')
        for test_file in tqdm(test_list_pics[:-2]):
            f.write(f'r\'{test_dir + test_file}\',\n')  # there will be 1 ',' too much!
        f.write(f'r\'{test_dir + test_list_pics[-1]}\'\n]')  # last without ','

        # Print validation-set
        f.write('\n\nval = [\n')
        for file in tqdm(val_list[:-2]):
            f.write(f'r\'{val_dir + file}\',\n')
        f.write(f'r\'{val_dir + val_list[-1]}\'\n]')

        # Print train-set
        f.write('\n\ntrain = [\n')
        for file in tqdm(train_list[:-2]):
            f.write(f'r\'{source_dir + file}\',\n')
        f.write(f'r\'{source_dir + train_list[-1]}\'\n]')

        print(f'Final values train {len(train_list)}, val {len(val_list)}, test {len(test_list_pics)}')
        print(f'Final prop train {len(train_list) / (len(train_list) + len(val_list) + len(test_list_pics))}, '
              f'val {len(val_list) / (len(train_list) + len(val_list) + len(test_list_pics))}, '
              f'test {len(test_list_pics) / (len(train_list) + len(val_list) + len(test_list_pics))}')
