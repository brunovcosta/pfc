import tensorflow as tf
import numpy as np
from sklearn import metrics
from keras.callbacks import TensorBoard


class TrainValTensorBoard(TensorBoard):
    def __init__(self, train_data, log_dir='./tf_logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        self.train_data = train_data
        training_log_dir = f'{log_dir}/training'
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = f'{log_dir}/validation'

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)

        metrics = Metrics(
            self.model,
            self.train_data,
            self.validation_data,
            self.val_writer,
            epoch)
        metrics.write_metrics()
        
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class Metrics:

    def __init__(self, model, train_data, validation_data, val_writer, epoch):
        self.predict_probability = {}
        self.predict_label = {}
        self.target_one_hot = {}
        self.target_label = {}
        for name, data in [('val', validation_data), ('train', train_data)]:
            self.predict_probability[name] = np.asarray(model.predict(data[0]))
            self.predict_label[name] = np.argmax(self.predict_probability[name], axis=1)
            self.target_one_hot[name] = np.asarray(data[1])
            self.target_label[name] = np.argmax(self.target_one_hot[name], axis=1)
        self.epoch = epoch
        self.model = model
        self.val_writer = val_writer

    def write_metrics(self):

        def calc_write(metric_func, target, predict):
            for name in ['val', 'train']:
                plot_name = f'{metric_func.__name__}_{name}'
                metric_value = metric_func(target[name], predict[name])
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = metric_value
                summary_value.tag = plot_name
                self.val_writer.add_summary(summary, self.epoch)

        metrics_list_label = [
            metrics.accuracy_score,
            self.precision_score,
            self.recall_score,
            self.f1_score
        ]
        for metric_func in metrics_list_label:
            calc_write(metric_func, self.target_label, self.predict_label)

        metrics_list_probability = [
            self.roc_auc_score
        ]
        for metric_func in metrics_list_probability:
            calc_write(metric_func, self.target_one_hot, self.predict_probability)

    def precision_score(self, target_label, predict_label):
        return metrics.precision_score(target_label, predict_label, average='micro')

    def recall_score(self, target_label, predict_label):
        return metrics.recall_score(target_label, predict_label, average='micro')

    def f1_score(self, target_label, predict_label):
        return metrics.f1_score(target_label, predict_label, average='micro')

    def roc_auc_score(self, target_label, predict_label):
        return metrics.roc_auc_score(target_label, predict_label, average='micro')
