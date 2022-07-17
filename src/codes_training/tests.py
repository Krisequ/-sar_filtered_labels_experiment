# TODO REMOVE?
import tensorflow as tf
import model_definitions
import random
import os
import numpy as np

import sequence_generator
import utils
import train_val_test_lists as data_lists
import sequence_generator as seq


# To read?
# https://www.techtarget.com/searchenterpriseai/feature/How-to-troubleshoot-8-common-autoencoder-limitations

test_name = 'bardzo_test'
images_directory = 'E:\\PobieranieESA\\CAPELLA_C02_SP_GEO_HH_20210310201217_20210310201243\\'
saving_dir = '../complete_backups/' + test_name + '/'

if not os.path.exists('../complete_backups/'):
    os.mkdir('../complete_backups')
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

random.seed(1133)
is_on_serv = utils.is_on_server()
if is_on_serv:  # GPU and directory setup
    print('Starting server setup')
    images_directory = '../'
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Changing OF THE SPLIT!!
    random.shuffle(data_lists.val)
    data_lists.train = data_lists.train + (data_lists.val[480:])
    data_lists.val = data_lists.val[:480]  # TODO
    # Changing OF THE SPLIT!!
else:
    print('Starting testing setup')
    data_lists.train = utils.replacement_list  # TODO remove
    data_lists.val = utils.replacement_list[:240]
print(f'Loaded image lists: train {len(data_lists.train)}, val {len(data_lists.val)} and test {len(data_lists.test)}')

data_list_for_show = data_lists.train[0:20]  # TODO to be changed - images for publication in article
random.shuffle(data_lists.train)

# Saved figures - display examples after sequence modification and setup for epoch-wise example saving
batch_x = seq.perform_sequence_test(data_list_for_show, images_directory)
print(f'Loaded batch_x for example images of shape {batch_x.shape}')






