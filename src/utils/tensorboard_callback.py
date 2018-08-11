import tensorflow as tf
from .metrics import MetricsTensorboard


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, train_data, log_dir='./logs/tf_logs', **kwargs):
        self.training_data = train_data
        # Make the original `TensorBoard` log to a subdirectory 'training'
        self.log_dir_train = f'{log_dir}/training'
        super().__init__(self.log_dir_train, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.log_dir_val = f'{log_dir}/validation'

    def set_model(self, model):
        # Setup writer
        self._writer = {}
        self._writer['train'] = tf.summary.FileWriter(self.log_dir_train)
        self._writer['val'] = tf.summary.FileWriter(self.log_dir_val)
        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        metrics = MetricsTensorboard(
            self.model,
            self.training_data,
            self.validation_data)
        metrics.write_metrics_to_tensorboard(
            self._writer,
            epoch)

        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        for name in ['val', 'train']:
            self._writer[name].close()
