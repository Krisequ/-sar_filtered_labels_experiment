import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5)


def display_cnn_weights(layer, path):
    if 'conv' not in layer.name:
        print(f'Skipping layer - layer: {layer}')
        return

    filters, biases = layer.get_weights()
    a_min, a_max = filters.min(), filters.max()

    cols, rows = 4, 4
    n_filters = cols*rows

    color = True
    if not color:
        filters_scaled = (filters - a_min) / (a_max - a_min)
        fig, axs = plt.subplots(rows, cols)
        fig.set_size_inches(rows, cols)
        for idx in range(min(n_filters, filters_scaled.shape[-1])):
            axs[idx//rows, idx % cols].axis('off')
            axs[idx//rows, idx % cols].imshow(filters_scaled[:, :, 0, idx], cmap='gray')
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.1, hspace=0.1)
        fig.savefig(fname=path + '.png', dpi=300, transparent=True, bbox_inches='tight')
    else:
        filters_scaled = filters.flatten()
        for idx, val in enumerate(filters_scaled):  # values -1 to 1 will do the best, with keeping 0 as 0
            if val < 0:
                filters_scaled[idx] = val / (-a_min)
            else:
                filters_scaled[idx] = val / a_max
        filters_scaled = filters_scaled.reshape(filters.shape)

        plt.figure(figsize=(rows*filters_scaled.shape[0], cols*filters_scaled.shape[1]))
        for idx in range(min(n_filters, filters_scaled.shape[-1])):
            plt.subplot(rows, cols, 1 + idx)
            sns.heatmap(filters_scaled[:, :, 0, idx], cmap="bwr", annot=filters_scaled[:, :, 0, idx],
                        cbar=False, vmin=-1, vmax=1, linewidths=1, linecolor=(0, 0, 0))
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.1, hspace=0.1)
        plt.savefig(fname=path + '.png', dpi=300, transparent=True, bbox_inches='tight')
        plt.close()
    return


def display_features(model, img, layer: int, path):
    img = np.reshape(img, (1, 1024, 1024, 1))
    feat_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[layer].output)
    feature_maps = feat_model.predict(img)
    print('Features shape'+str(feature_maps.shape))

    depth = feature_maps.shape[-1]
    cols, rows = 4, 4  # max values
    n_filters = cols*rows

    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(rows*2, cols*2)
    for idx in range(min(n_filters, depth)):
        axs[idx // rows, idx % cols].axis('off')
        axs[idx // rows, idx % cols].imshow(feature_maps[0, :, :, idx], cmap='gray')

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.1, hspace=0.1)
    fig.savefig(fname=path + '.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    return
