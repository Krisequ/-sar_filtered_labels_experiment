import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import sequence_generator


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def ssim_mae_loss(y_true, y_pred) -> float:
    return np.float32(0.5*ssim_loss(y_true, y_pred)+0.5*tf.keras.metrics.mean_absolute_error(y_true, y_pred))


def get_tb_callback(prefix: str = 'training'):
    """
    Get tensorboard callback - with date and time file naming
    """

    date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = f'../logs/{prefix}-{date}'
    print(f'Starting TensorBoard callback {log_dir}')
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def get_check_point_callback(filepath: str, save_freq: str = "epoch",
                             save_best_only: bool = False, monitor="val_accuracy",
                             mode='max'):
    return tf.keras.callbacks.ModelCheckpoint(filepath,
                                              monitor=monitor,
                                              verbose=1,
                                              save_best_only=save_best_only,
                                              save_weights_only=False,
                                              mode=mode,
                                              save_freq=save_freq,
                                              options=None,
                                              )


class ExampleFigureCallback(tf.keras.callbacks.Callback):
    """
    Callback that creates images with inputs and results of the inputs
    """
    def __init__(self,  batch_x, save_dir='./end_epoch_results/'):
        self.save_dir = save_dir
        self.batch_x = batch_x
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):

        try:  # Tested
            print('Starting example figure generation')
            original_shape = self.batch_x.shape
            batch_y = np.zeros(self.batch_x.shape)
            for idx in range(len(self.batch_x)):
                batch_y[idx] = self.model.predict(self.batch_x[idx].reshape(1, original_shape[1], original_shape[2],
                                                                            original_shape[3]))
            sequence_generator.display_example([[self.batch_x, batch_y]],
                                               title=self.save_dir+'epoch'+str(epoch))
        except Exception as e:
            print(f'Couldn\'t print example, skipping error: {e}')
            print(f'batch_x shape: {self.batch_x.shape()}')
        print('Saved example figures')
        return
