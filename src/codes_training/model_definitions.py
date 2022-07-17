import tensorflow as tf


def get_cnn_autoencoder_model(input_shape=(1024, 1024, 1),
                              model_name: str = 'fully_CNN'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (5, 5), activation="relu", strides=2, padding="same", input_shape=input_shape),
        # 512x512x16     - 4x original data count
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 256x256x16   - 1x original data count
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),  # 256x256x32 - 2x odc
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 128x128x32 - 0.5 odc
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),  # # 256x256x32 - 1x odc
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 64*64*64   - final features - 0.25 odc
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same"),  # 128x128x128
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same"),  # 256x256x64
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),  # 512x512x32
        tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same"),  # 1024x1024x16
        tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")  # 1024x1024x1
    ])
    model._name = model_name
    return model


def get_compressing_cnn_autoencoder_model(input_shape=(1024, 1024, 1),
                                          model_name: str = 'compressing_CNN'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (5, 5), activation="relu", strides=2, padding="same", input_shape=input_shape),
        # 512 x 512 x 8     - 2x original data count
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 256x256x8 - 0.5x original data count
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),  # 256x256x16 - 0.5x odc
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 128x128x16 - 0.25 odc
        # tf.keras.layers.dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),  # 128x128x32 - 0.25x odc
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 64x64x32   - 0.125 odc
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),  # 64x64x64 - 0.25x odc
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 32x32x64   - 1/16 odc
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),  # 32x32x128 1/8 odc
        tf.keras.layers.MaxPooling2D((2, 2), padding="same"),  # 16x16x128   - final features - 1/32 odc
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same"),  # 32x32x128
        tf.keras.layers.Conv2DTranspose(96, (3, 3), strides=2, activation="relu", padding="same"),  # 64x64x64
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same"),  # 128x128x64
        tf.keras.layers.Conv2DTranspose(48, (3, 3), strides=2, activation="relu", padding="same"),  # 256x256x48
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),  # 512x512x32
        tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same"),  # 1024x1024x16
        tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")  # 1024x1024x1
    ])
    model._name = model_name
    return model


def get_mixed_autoencoder_model(input_shape=(1024, 1024, 1),
                                model_name: str = 'compressing_CNN'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(4, (3, 3), strides=2, padding='same', activation='relu', input_shape=input_shape),
        # 512x512x2 - 0.5odc
        tf.keras.layers.MaxPooling2D((2, 2)),  # 256x256x2 - 0.25 odc
        tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=input_shape),  # 256x256x4
        tf.keras.layers.MaxPooling2D((2, 2)),  # 128x128x4 - 1/16 odc
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),  # 64x64x8
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling2D((2, 2)),  # 64x64x16 - 1/32 odc
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),  # 32x32x16
        tf.keras.layers.MaxPooling2D((2, 2)),  # 32x32x32 - 1/64 odc
        tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding='same', activation='relu'),  # 16x16x32
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(128, activation='softmax'),
        tf.keras.layers.Dense(512, activation='softmax'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='softmax'),
        tf.keras.layers.Dense(4096, activation='softmax'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Reshape((32, 32, 4)),
        # tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),  # 32x32x128
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),  # 64x64x64
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),  # 128x128x64
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same'),  # 256x256x48
        tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same'),  # 512x512x32
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(4, (5, 5), strides=2, activation='relu', padding='same'),  # 1024x1024x4
        tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')])  # 1024x1024x1

    model._name = model_name
    return model


def get_vgg19_backbone_model(input_shape=(1024, 1024, 1),
                             model_name: str = 'VGG19_backbone'):  # for RCNN

    #  https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
    model = tf.keras.models.Sequential([  # VGG19
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),  # end of VGG19


        tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=2, activation='relu', padding='same'),  # 64x64x256
        tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same'),  # 128x128x125
        tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),  # 256x256x64
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),  # 512x512x16
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),  # 1024x1024x8
        tf.keras.layers.Conv2D(16, (3, 3), activation='sigmoid', padding='same'),  # 1024 x 1024 x 1
        tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # 1024 x 1024 x 1
    ])

    model._name = model_name
    return model


def get_vgg19_backbone_model_batched(input_shape=(1024, 1024, 1),
                                     model_name: str = 'VGG19_backbone_patched'):  # for RCNN

    model = tf.keras.models.Sequential([
        tf.keras.applications.VGG19(include_top=False,
                                    weights=None,
                                    input_shape=input_shape,
                                    pooling=None,
                                    classes=1024
                                    ),  # outputs 32, 32, 512
        tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same'),  # 64x64x256
        tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),  # 128x128x125
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),  # 256x256x64
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same'),  # 512x512x16
        tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same'),  # 1024x1024x8
        tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # 1024 x 1024 x 1
    ])
    model._name = model_name
    return model


def get_vgg16_backbone_model_batched(input_shape=(1024, 1024, 1),
                                     model_name: str = 'VGG16_backbone_patched'):  # for RCNN

    model = tf.keras.models.Sequential([
        tf.keras.applications.VGG16(include_top=False,
                                    weights=None,
                                    input_shape=input_shape,
                                    pooling=None,
                                    classes=1024
                                    ),  # outputs 32, 32, 512
        tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same'),  # 64x64x256
        tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),  # 128x128x125
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),  # 256x256x64
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same'),  # 512x512x16
        tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same'),  # 1024x1024x8
        tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # 1024 x 1024 x 1
    ])
    model._name = model_name
    return model

# def get_transformer_model():  #  TODO
#     return
