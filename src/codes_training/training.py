import tensorflow as tf
from PIL import Image
import numpy as np
import random
import time
import numpy as np
import os

import sequence_generator as seq
import train_val_test_lists as data_lists
import model_definitions
import utils

# TODO Try L1 and L2, KL divergence loss
# TODO Pruning
#  "odszumianie", L1, L2, KL?


for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
random.seed(1133)

images_directory = ''

print(f'Loaded image lists: train {len(data_lists.train)}, val {len(data_lists.val)} '
      f'and test {len(data_lists.test)}')

data_list_for_show = data_lists.train[0:20]  # TODO to be changed - images for publication in article
random.shuffle(data_lists.train)
# Saved figures - display examples after sequence modification and setup for epoch-wise example saving
seq.perform_sequence_test(data_list_for_show, images_directory, target_directory='../outputs/')


def train(autoencoder,
          generator_variant: seq.SeqTypes = seq.SeqTypes.simple,
          batch_size: int = 32,
          initial_epoch: int = 0  # for resuming of the training
          ):

    test_name = autoencoder._name
    input_shape = autoencoder.layers[0].input.shape
    input_shape = (input_shape[1], input_shape[2], input_shape[3])
    saving_dir = '../complete_backups/' + test_name + '/'

    if not os.path.exists('../complete_backups/'):
        os.mkdir('../complete_backups')
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    batch_x = np.array([np.asarray(Image.open(os.path.join(images_directory, f)).
                                   resize((input_shape[0], input_shape[1]))) for f in data_list_for_show])
    batch_x = batch_x.astype(np.float32) / 255.0
    batch_x = np.reshape(batch_x, (len(batch_x), input_shape[0], input_shape[1], input_shape[2]))
    print(f'Loaded batch_x for example images of shape {batch_x.shape}')

    # Training setup
    lr_start = 1e-1
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_start)
    # loss_func = utils.ssim_mae_loss
    # loss_func = utils.ssim_loss
    # loss_func = 'mae'  # dont want to focus on single outlier pixels
    loss_func = 'mse'
    metrics = ['accuracy', 'mse', 'mae']  # , utils.ssim_loss
    epochs = 180
    sub_epochs = 120  # 120

    autoencoder.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    autoencoder.summary()

    # noinspection PyUnusedLocal
    def lr_scheduler(epoch, lr):
        curr_lr = lr_start
        if epoch > 5:
            curr_lr = tf.math.maximum(lr_start * tf.math.exp(-0.04*(epoch-5)), 1e-4)
        print(f'Learning rate set to {curr_lr}')
        return curr_lr

    # batch_y = autoencoder.predict(batch_x)
    # seq.display_example([[batch_x, batch_y]], title='./starting_point')

    if not os.path.exists(saving_dir+'end_of_epoch_predict_autoencoder/'):
        os.mkdir(saving_dir+'end_of_epoch_predict_autoencoder/')

    monitor_for_callbacks = 'val_loss'
    mode = 'min'
    callbacks = [
                utils.get_tb_callback(saving_dir + 'log_' + test_name),  # saving in backup
                utils.get_check_point_callback(saving_dir + 'model_autosave',
                                               save_freq='epoch',
                                               save_best_only=False),
                utils.get_check_point_callback(saving_dir + 'model_best',
                                               save_freq='epoch',
                                               save_best_only=True,
                                               monitor=monitor_for_callbacks,
                                               mode=mode
                                               ),
                tf.keras.callbacks.EarlyStopping(patience=30,
                                                 monitor=monitor_for_callbacks,
                                                 mode=mode
                                                 ),
                utils.ExampleFigureCallback(batch_x, save_dir=saving_dir + 'end_of_epoch_predict_autoencoder/'),
                tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
                ]

    print('Starting learning in 2 seconds')
    time.sleep(2)

    interrupted = False
    try:
        autoencoder.fit(seq.get_sequence_by_type(sequence_type=generator_variant,
                                                 files_list=data_lists.train,
                                                 batch_size=batch_size,
                                                 source_dir=images_directory,
                                                 sub_epochs=sub_epochs,
                                                 shape=input_shape),
                        validation_data=seq.get_sequence_by_type(sequence_type=generator_variant,
                                                                 files_list=data_lists.val,
                                                                 batch_size=batch_size,
                                                                 source_dir=images_directory,
                                                                 sub_epochs=1,
                                                                 shape=input_shape),
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks)

        saving_dir = saving_dir + 'model_final'
    except KeyboardInterrupt:
        interrupted = True
        saving_dir = saving_dir + 'model_interrupted'
        print(f'Learning interrupted, changing saving directory to {saving_dir}')

    autoencoder.save(saving_dir)
    print(f'Saved weights to {saving_dir}')

    # seq_gen = seq.simple_image_sequence(data_lists.train, 20, images_directory)
    predictions = autoencoder.predict(batch_x)
    seq.display_example([[batch_x, predictions]], title=test_name+'outputs_after_learning', t_dir=saving_dir)

    if interrupted:
        raise KeyboardInterrupt('Interrupting process')  # Close whole training cycle
    return


learning_queue = np.array([
{  # 0
 'model': model_definitions.get_cnn_autoencoder_model(model_name='fullyCNN_trained_without_filters'),
 'generator_variant': seq.SeqTypes.simple,
 'batch_size': 32,
 'initial_epoch': 0},
{  # 1
 'model': model_definitions.get_cnn_autoencoder_model(model_name='fullyCNN_trained_with_7x7_filters'),
 'generator_variant': seq.SeqTypes.filtered,
 'batch_size': 32,
 'initial_epoch': 0},

{  # 2
 'model': model_definitions.get_compressing_cnn_autoencoder_model(model_name='compressing_CNN_trained_without_filters'),
 'generator_variant': seq.SeqTypes.simple,
 'batch_size': 32,
 'initial_epoch': 0},
{  # 3
 'model': model_definitions.get_compressing_cnn_autoencoder_model(model_name='compressing_CNN_trained_with_7x7_filters'),
 'generator_variant': seq.SeqTypes.filtered,
 'batch_size': 32,
 'initial_epoch': 0},

{  # 4
 'model': model_definitions.get_mixed_autoencoder_model(model_name='mixed_dense_CNN_trained_without_filters'),
 'generator_variant': seq.SeqTypes.simple,
 'batch_size': 128,
 'initial_epoch': 0},
{  # 5
 'model': model_definitions.get_mixed_autoencoder_model(model_name='mixed_dense_CNN_trained_with_7x7_filters'),
 'generator_variant': seq.SeqTypes.filtered,
 'batch_size': 128,
 'initial_epoch': 0},

{  # 6
 'model': model_definitions.get_vgg19_backbone_model(model_name='VGG19_backbone_without_filters'),
 'generator_variant': seq.SeqTypes.simple,
 'batch_size': 6,
 'initial_epoch': 0},
{  # 7
 'model': model_definitions.get_vgg19_backbone_model(model_name='VGG19_backbone_with_7x7_filters'),
 'generator_variant': seq.SeqTypes.filtered,
 'batch_size': 6,
 'initial_epoch': 0},

{  # 8
 'model': model_definitions.get_vgg16_backbone_model_batched(model_name='VGG16_backbone_without_filters'),
 'generator_variant': seq.SeqTypes.simple,
 'batch_size': 8,
 'initial_epoch': 0},
{  # 9
 'model': model_definitions.get_vgg16_backbone_model_batched(model_name='VGG16_backbone_with_7x7_filters'),
 'generator_variant': seq.SeqTypes.filtered,
 'batch_size': 8,
 'initial_epoch': 0},
{  # 10
 'model': model_definitions.get_vgg19_backbone_model(input_shape=(224, 224, 1),
                                                     model_name='VGG19_backbone_without_filters_224x224x1'),
 'generator_variant': seq.SeqTypes.simple,
 'batch_size': 6,
 'initial_epoch': 0},
{  # 11
 'model': model_definitions.get_vgg19_backbone_model(input_shape=(224, 224, 1),
                                                     model_name='VGG19_backbone_with_7x7_filters_224x224x1'),
 'generator_variant': seq.SeqTypes.filtered,
 'batch_size': 6,
 'initial_epoch': 0},
    # TODO ADD filtered input filtered output

# {  # 10
#  'model': tf.keras.models.load_model(
#      '../complete_backups/VGG19_backbone_without_filters/model_interrupted_'),
#  'generator_variant': seq.SeqTypes.simple,
#  'batch_size': 6,
#  'initial_epoch': 8},
                ])

for params in learning_queue[[9]]:
    print('\n\n\n\n\n Starting model training')
    train(autoencoder=params['model'], generator_variant=params['generator_variant'],
          batch_size=params['batch_size'], initial_epoch=params['initial_epoch'])
