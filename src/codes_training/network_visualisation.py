from PIL import Image
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
from keras.utils.generic_utils import get_custom_objects

import train_val_test_lists as data_lists
import sequence_generator as seq
import utils
import network_visualization_lib as vis

# TODO Violin plot - CPU vs GPU times for each network?

models = [
    # {'dir': '../complete_backups/compressing_CNN_trained_without_filters/model_best',
    #  'name': 'compressing_CNN_trained_without_filters', 'test_layers': []},  # test all or [0, 1, 2, 3, 4, 5, 6]
    # {'dir': '../complete_backups/compressing_CNN_trained_without_filters_with_SSIM/model_best',
    #  'name': 'compressing_CNN_trained_without_filters_with_SSIM', 'test_layers': []},
    # {'dir': '../complete_backups/compressing_CNN_trained_with_7x7_filters/model_best',
    #  'name': 'compressing_CNN_trained_with_7x7_filters', 'test_layers': []},
    # {'dir': '../complete_backups/compressing_CNN_trained_with_7x7_filters_with_SSIM/model_best',
    #  'name': 'compressing_CNN_trained_with_7x7_filters_with_SSIM', 'test_layers': []},
    # {'dir': '../complete_backups/fullyCNN_trained_without_filters/model_best',
    #  'name': 'fullyCNN_trained_without_filters', 'test_layers': []},
    # {'dir': '../complete_backups/fullyCNN_trained_with_7x7_filters/model_best',
    #  'name': 'fullyCNN_trained_with_7x7_filters', 'test_layers': []},
    # {'dir': '../complete_backups/mixed_dense_CNN_trained_without_filters/model_best',
    #  'name': 'mixed_dense_CNN_trained_without_filters', 'test_layers': []},
    # {'dir': '../complete_backups/mixed_dense_CNN_trained_with_7x7_filters/model_best',
    #  'name': 'mixed_dense_CNN_trained_with_7x7_filters', 'test_layers': []},
    # {'dir': '../complete_backups/VGG16_backbone_without_filters/model_best',
    #  'name': 'VGG16_backbone_without_filters', 'test_layers': []},
    # {'dir': '../complete_backups/VGG16_backbone_with_7x7_filters/model_best',
    #  'name': 'VGG16_backbone_with_7x7_filters', 'test_layers': []},
    # {'dir': '../complete_backups/VGG19_backbone_without_filters/model_best',
    #  'name': 'VGG19_backbone_without_filters', 'test_layers': []},
    # {'dir': '../complete_backups/VGG19_backbone_with_7x7_filters/model_best',
    #  'name': 'VGG19_backbone_with_7x7_filters', 'test_layers': []},
    {'dir': '../outputs/complete_backups/compressing_CNN_trained_with_7x7_filters/model_best',
     'name': 'compressing_CNN_trained_with_7x7_filters', 'test_layers': []},
]

# example_images = []

test_images = data_lists.test
example_images = test_images[:640]  # TODO replace
# example_images = test_images[:64]  # TODO replace
images_directory = '../'

batch_x = np.array([np.asarray(Image.open(os.path.join(images_directory, f))) for f in example_images])
batch_x = batch_x.astype(np.float32) / 255.0
batch_x = np.reshape(batch_x, (len(batch_x), 1024, 1024, 1))
print(f'Loaded batch_x for example images of shape {batch_x.shape}')


new_batch_x = np.array([np.asarray(Image.open(os.path.join(images_directory, f))) for f in test_images[:100]])
new_batch_x = new_batch_x.astype(np.float32) / 255.0
new_batch_x = np.reshape(new_batch_x, (len(new_batch_x), 1, 1024, 1024, 1))
print(f'Loaded batch_x for inference times of shape {new_batch_x.shape}')


def create_dir(dir_loc: str) -> str:
    if not os.path.exists(dir_loc):
        os.mkdir(dir_loc)
    return dir_loc


create_dir('../outputs/visualisation')
x_y_batch_comparison_dir = create_dir('../outputs/visualisation/inputs_outputs')
layer_visualisation_dir = create_dir('../outputs/visualisation/layer')
feature_visualisation_dir = create_dir('../outputs/visualisation/feature')


get_custom_objects().update({"ssim_loss": utils.ssim_loss})
metrics = ['accuracy', 'mse', 'mae']

# tabular_output = pd.DataFrame(index=[mod['name'] for mod in models])
tabular_output = []

for mod in models:
    print(f"\n\n Reading from {mod['dir']} - name {mod['name']}")
    try:
        # custom_objects = {'ssim_loss': utils.ssim_loss}
        model = tf.keras.models.load_model(mod['dir'])  # ,custom_objects=custom_objects)  # loader
        model.compile(optimizer='adam', loss='mae', metrics=metrics)
        model.summary()
        tabular_output.append({'Dir': mod['dir'], 'Name': model._name, 'Neural Network': mod['name']})

        print(f"Generating example outputs")
        batch_y = np.zeros(batch_x.shape)
        for idx in range(len(batch_x)):
            batch_y[idx] = model.predict(batch_x[idx].reshape(1, 1024, 1024, 1))
        seq.display_example([[batch_x, batch_y]],
                            title='Inputs versus outputs for '+mod['name'],
                            t_dir='../outputs/visualisation/inputs_outputs/')
        # TODO display differences?

        print(f"Performing inference time test")
        start_time = time.time()  # Tic
        for batch in new_batch_x:
            model.predict(batch)
        end_time = time.time()  # TOC
        print(f"Inference of {len(new_batch_x)} images took {end_time - start_time}"
              f" - at average {(end_time - start_time)/len(new_batch_x)} s/image")
        tabular_output[-1]['Avg image time'] = (end_time - start_time)/len(new_batch_x)

        print(f"Performing metrics tests")
        unfiltered = model.evaluate(seq.get_sequence_by_type(seq.SeqTypes.simple,
                                                             files_list=test_images[0:10],
                                                             batch_size=16,
                                                             source_dir='../',
                                                             sub_epochs=1),
                                    verbose=1)

        filtered = model.evaluate(seq.get_sequence_by_type(seq.SeqTypes.filtered,
                                                           files_list=test_images[0:10],
                                                           batch_size=16,
                                                           source_dir='../',
                                                           sub_epochs=1),
                                  verbose=1)

        noised = model.evaluate(seq.get_sequence_by_type(seq.SeqTypes.noised,
                                                         files_list=test_images[0:10],
                                                         batch_size=16,
                                                         source_dir='../',
                                                         sub_epochs=1),
                                verbose=1)
        noised_x_filtered_y = model.evaluate(seq.get_sequence_by_type(seq.SeqTypes.noised,
                                                                      files_list=test_images[0:10],
                                                                      batch_size=16,
                                                                      source_dir='../',
                                                                      sub_epochs=1),
                                             verbose=1)

        # 0 is loss function = mae
        print('\nUnfiltered ', end='')
        for idx in range(len(metrics)):
            dict_idx = 'Unfiltered '+metrics[idx]
            tabular_output[-1][dict_idx] = unfiltered[idx+1]  # skipping loss val
            print(f'{metrics[idx]} {unfiltered[idx+1]}', end='\t')

        print('\nFiltered ', end='')
        for idx in range(len(metrics)):
            dict_idx = 'Filtered ' + metrics[idx]
            tabular_output[-1][dict_idx] = filtered[idx + 1]  # skipping loss val
            print(f'{metrics[idx]} {filtered[idx + 1]}', end='\t')

        print('\nNoised ', end='')
        for idx in range(len(metrics)):
            dict_idx = 'Noised ' + metrics[idx]
            tabular_output[-1][dict_idx] = noised[idx + 1]  # skipping loss val
            print(f'{metrics[idx]} {noised[idx + 1]}', end='\t')

        print('\nNoised and filtered ', end='')
        for idx in range(len(metrics)):
            dict_idx = 'Noised x with filtered y ' + metrics[idx]
            tabular_output[-1][dict_idx] = noised_x_filtered_y[idx + 1]  # skipping loss val
            print(f'{metrics[idx]} {noised_x_filtered_y[idx + 1]}', end='\t')

        # print(f"Unfiltered: {metrics[0]} {unfiltered[1]}, {metrics[1]} {unfiltered[2]}, {metrics[2]} {unfiltered[3]}")
        # print(f"Filtered: {metrics[0]} {filtered[1]}, {metrics[1]} {filtered[2]}, {metrics[2]} {filtered[3]}")
        # # print(f"Filtered: accuracy {filtered['accuracy']}, binary_crossentropy {filtered['binary_crossentropy']},
        # #       mse {filtered['mse']}, mae {filtered['mae']}")

        # print(f"Creating random artificial images") #TODO
        # print(f"Creating artificial images based on image") #TODO

        print(f"Starting layer analysis")
        for idx, layer in enumerate(model.layers):
            directory = layer_visualisation_dir + '/' + mod['name'] + '_layer_' + str(idx)

            if hasattr(layer, 'layers'):  # if functional_print insides
                for d_idx, d_layer in enumerate(layer.layers):
                    d_dir = directory + '_' + str(d_idx)
                    vis.display_cnn_weights(layer, d_dir)
                    print(f"Saving Convolutional layer analysis to {d_dir}")
            else:
                vis.display_cnn_weights(layer, directory)
                print(f"Saving Convolutional layer analysis to {directory}")

        layers_to_check = mod['test_layers']
        if len(layers_to_check) == 0:
            layers_to_check = [i for i in range(len(model.layers))]
        print(f'Layers to extract features from: {layers_to_check}')

        for layer_idx in layers_to_check:
            directory = feature_visualisation_dir + '/' + mod['name'] + '_feature_layer_' + str(layer_idx)
            temp = vis.display_features(model, batch_x[6, :, :, :], layer_idx, directory)  # TODO replace pic
            print(f"Saving feature layer analysis to {directory}")

    except Exception as e:  # for invalid model directory
        print(mod['name']+' - '+str(e))
        continue


print('Saving table')
print(tabular_output)
tab = pd.DataFrame.from_dict(tabular_output)
tab.to_csv('Output.csv', sep=';')

print('\n Finished')
