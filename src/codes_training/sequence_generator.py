import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from enum import Enum


class SeqTypes(Enum):
    simple = 1
    filtered = 2
    noised = 3
    noised_x_and_filtered_y = 4


def get_sequence_by_type(sequence_type: SeqTypes, files_list, batch_size, source_dir, sub_epochs=12, dtype=np.float32,
                         shape=(1024, 1024, 1), suppress_aug: bool = False):
    """ Universal getter to all of the sequence manager classes """

    if sequence_type == SeqTypes.simple:
        print(f'Starting simple sequence with {len(files_list)} images, batch_size {batch_size} '
              f'and {sub_epochs} sub-epochs')
        return SimpleImageSequence(files_list, batch_size=batch_size, source_dir=source_dir, sub_epochs=sub_epochs,
                                   dtype=dtype, shape=shape, suppress_aug=suppress_aug)
    elif sequence_type == SeqTypes.filtered:
        print(f'Starting filtered sequence with {len(files_list)} images, batch_size {batch_size} '
              f'and {sub_epochs} sub-epochs')
        return MedianFilteredOutputImageSequence(files_list, batch_size=batch_size, source_dir=source_dir,
                                                 sub_epochs=sub_epochs, kernel_size=7, dtype=dtype, shape=shape,
                                                 suppress_aug=suppress_aug)
    elif sequence_type == SeqTypes.noised:
        print(f'Starting noised sequence with {len(files_list)} images, batch_size {batch_size} '
              f'and {sub_epochs} sub-epochs')
        return GaussianNoiseImageSequence(files_list, batch_size=batch_size, source_dir=source_dir,
                                          sub_epochs=sub_epochs, mean=0, sigma=50, dtype=dtype, shape=shape,
                                          suppress_aug=suppress_aug)
    elif sequence_type == SeqTypes.noised_x_and_filtered_y:
        print(f'Starting noised sequence with {len(files_list)} images, batch_size {batch_size} '
              f'and {sub_epochs} sub-epochs')
        return GaussianNoiseWithFilteredOutputImageSequence(files_list, batch_size=batch_size, source_dir=source_dir,
                                                            sub_epochs=sub_epochs, kernel_size=7, mean=0, sigma=50,
                                                            dtype=dtype, shape=shape, suppress_aug=suppress_aug)
    else:
        print('NOT IMPLEMENTED')
        return None


def display_example(sequence, plot_cols: int = 8, plot_rows: int = 2, title: str = 'ExampleDisplay', t_dir: str = './'):
    """ Function used in end_of_epoch predictions """
    for im_seq in sequence:
        plt.suptitle(title)
        for i in range(plot_cols*plot_rows):
            row = int(2*(np.floor(i/plot_cols)))
            col = int(i % plot_cols)
            plt.subplot(plot_rows * 2, plot_cols,  int(1 + col+plot_cols*row))
            plt.imshow(im_seq[0][i].astype(np.float32),
                       cmap=plt.cm.get_cmap('gray'), vmin=0.0, vmax=1.0)
            # plt.title(i)
            plt.axis('off')
            plt.subplot(plot_rows * 2, plot_cols, int(1 + col+plot_cols*(row+1)))
            plt.imshow(im_seq[1][i].astype(np.float32),
                       cmap=plt.cm.get_cmap('gray'), vmin=0.0, vmax=1.0)
            # plt.title(i)
            plt.axis('off')
        break
    # plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.5)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.1, hspace=0.1)
    print("Saving to"+t_dir+title+'.png')
    plt.savefig(fname=t_dir+title+'.png', dpi=300, transparent=True, bbox_inches='tight')
    return


# noinspection PyTypeChecker
def perform_sequence_test(data_list_for_show: [str], images_directory: str,  target_directory: str = './') -> np.array:
    """ Function creating 3 sequences and writing them to files.
        Returns batch_x for further processing (epoch predictions """
    seq_gen = get_sequence_by_type(SeqTypes.simple, data_list_for_show, 20, images_directory, suppress_aug=True)
    display_example(seq_gen, title='Basic sequence', t_dir=target_directory)

    seq_gen = MedianFilteredOutputImageSequence(data_list_for_show, 20, images_directory, kernel_size=15,
                                                suppress_aug=True)
    display_example(seq_gen, title='Sequence with Filtered y', t_dir=target_directory)

    seq_gen = get_sequence_by_type(SeqTypes.noised, data_list_for_show, 20, images_directory, suppress_aug=True)
    display_example(seq_gen, title='Sequence with noised X', t_dir=target_directory)

    seq_gen = get_sequence_by_type(SeqTypes.noised_x_and_filtered_y, data_list_for_show, 20, images_directory,
                                   suppress_aug=True)
    display_example(seq_gen, title='Sequence with noised X and filtered y', t_dir=target_directory)
    del seq_gen
    # batch_y = autoencoder.predict(batch_x)
    # seq.display_example([[batch_x,batch_y]])
    return


def random_rot_mir(batch_x):
    """ Function performing random rotations and mirroring - for batch_x"""
    rotation = random.randrange(4)  # random <0 - 3> where 0 is no rotation
    batch_x = [rotate_by_90(x, rotation) for x in batch_x]

    mirroring = random.randrange(4)  # random <0 - 3> where 0 is no mirroring
    batch_x = [mirror_by(x, mirroring) for x in batch_x]
    return np.array(batch_x)


def mirror_by(img: np.array, mirror_type: int = 1) -> np.array:
    if mirror_type == 1:
        return np.flip(img, 0)
    elif mirror_type == 2:
        return np.flip(img, 1)
    elif mirror_type == 3:
        return np.flip(np.flip(img, 0), 1)
    else:
        return img


def rotate_by_90(img: np.array, num_of_rotation: int = 1) -> np.array:
    if num_of_rotation < 4:
        return np.rot90(img, k=num_of_rotation, axes=(0, 1))
    raise ValueError


class SimpleImageSequence(tf.keras.utils.Sequence):
    def __init__(self, files_list, batch_size, source_dir, sub_epochs=12, dtype=np.float32, shape=(1024, 1024, 1),
                 suppress_aug: bool = False):
        self.files_list = files_list
        self.batch_size = batch_size
        self.source_dir = source_dir
        self.dtype = dtype
        self.sub_epochs = sub_epochs
        self.epoch = 0
        self.im_shape = shape
        self.suppress_aug = suppress_aug

    def __len__(self):
        return int(np.ceil(len(self.files_list) / self.batch_size / self.sub_epochs))

    def on_epoch_end(self):
        random.shuffle(self.files_list)
        self.epoch = (self.epoch + 1) % self.sub_epochs
        return

    def image_loading(self, idx):
        batch_x = []
        epoch_len = int(np.floor(len(self.files_list) / self.sub_epochs))

        for i in range(3):  # for exception handling - retry 3 times
            batch_files = self.files_list[(idx + i % self.__len__()) * self.batch_size + epoch_len * self.epoch:
                                          (1 + idx + i % self.__len__()) * self.batch_size + epoch_len * self.epoch]
            try:
                # batch_x = [np.asarray(Image.open(os.path.join(self.source_dir, f))) for f in batch_files]
                batch_x = [np.asarray(Image.open(os.path.join(self.source_dir, f)).
                                      resize((self.im_shape[0], self.im_shape[1]))) for f in batch_files]
                # batch_x = [np.asarray(cv2.resize(cv2.imread(os.path.join(self.source_dir, f)),
                #                                  dsize=(1024,1024))) for f in batch_files]
            except Exception as e:
                print('Problem with image loading:', e)

        if not self.suppress_aug:
            batch_x = random_rot_mir(batch_x)
        return batch_x

    def normalize_array(self, batch):
        batch = np.asarray(batch)
        batch = batch.astype(self.dtype) / 255.0
        batch = np.reshape(batch, (len(batch), self.im_shape[0], self.im_shape[1], self.im_shape[2]))
        return batch

    def __getitem__(self, idx):
        batch_x = self.normalize_array(self.image_loading(idx=idx))
        return batch_x, batch_x


class MedianFilteredOutputImageSequence(SimpleImageSequence):
    def __init__(self, files_list, batch_size, source_dir, sub_epochs=12, dtype=np.float32, kernel_size=5,
                 shape=(1024, 1024, 1), suppress_aug: bool = False):
        super().__init__(files_list, batch_size, source_dir, sub_epochs=sub_epochs, dtype=dtype, shape=shape,
                         suppress_aug=suppress_aug)
        self.kernel_size = kernel_size
        return

    def __getitem__(self, idx):
        batch_x = self.image_loading(idx)
        batch_y = [cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), 0) for x in batch_x]
        batch_y = [cv2.medianBlur(y, self.kernel_size) for y in batch_y]

        batch_x = self.normalize_array(batch_x)
        batch_y = self.normalize_array(batch_y)
        return batch_x, batch_y


class GaussianNoiseImageSequence(SimpleImageSequence):
    """ Adding Gaussian Noise to the batch X"""
    def __init__(self, files_list, batch_size, source_dir, sub_epochs=12, dtype=np.float32, mean=0, sigma=10,
                 shape=(1024, 1024, 1), suppress_aug: bool = False):
        super().__init__(files_list, batch_size, source_dir, sub_epochs=sub_epochs, dtype=dtype, shape=shape,
                         suppress_aug=suppress_aug)
        self.mean = mean
        self.sigma = sigma
        return

    def __getitem__(self, idx):
        batch_y = self.image_loading(idx)
        batch_y = self.normalize_array(batch_y)

        noise = np.random.normal(self.mean, self.sigma,
                                 [1, self.im_shape[0], self.im_shape[1], self.im_shape[2]]) / 255.
        batch_x = np.array(batch_y)
        for idx in range(len(batch_y)):
            batch_x[idx] = batch_y[idx].reshape(1, self.im_shape[0], self.im_shape[1], self.im_shape[2]) + noise

        assert batch_x.dtype == self.dtype
        assert batch_x.shape == batch_y.shape
        return batch_x, batch_y


class GaussianNoiseWithFilteredOutputImageSequence(SimpleImageSequence):
    """ Adding Gaussian Noise to the batch X"""
    def __init__(self, files_list, batch_size, source_dir, sub_epochs=12, dtype=np.float32, mean=0, sigma=10,
                 shape=(1024, 1024, 1), suppress_aug: bool = False, kernel_size=7):
        super().__init__(files_list, batch_size, source_dir, sub_epochs=sub_epochs, dtype=dtype, shape=shape,
                         suppress_aug=suppress_aug)
        self.mean = mean
        self.sigma = sigma
        self.kernel_size = kernel_size
        return

    def __getitem__(self, idx):
        batch_x = self.image_loading(idx)
        batch_y = [cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), 0) for x in batch_x]
        batch_y = [cv2.medianBlur(y, self.kernel_size) for y in batch_y]
        batch_y = self.normalize_array(batch_y)

        batch_x = self.normalize_array(batch_x)
        noise = np.random.normal(self.mean, self.sigma,
                                 [1, self.im_shape[0], self.im_shape[1], self.im_shape[2]]) / 255.
        # batch_x = np.array(batch_y)
        for idx in range(len(batch_y)):
            batch_x[idx] = batch_x[idx].reshape(1, self.im_shape[0], self.im_shape[1], self.im_shape[2]) + noise



        assert batch_x.dtype == self.dtype
        assert batch_x.shape == batch_y.shape
        return batch_x, batch_y
