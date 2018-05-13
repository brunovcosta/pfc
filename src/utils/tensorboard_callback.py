import tensorflow as tf
import numpy as np
from sklearn import metrics


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, train_data, log_dir='./tf_logs', **kwargs):
        self.training_data = train_data
        # Make the original `TensorBoard` log to a subdirectory 'training'
        self.log_dir_train = f'{log_dir}/training'
        super(TrainValTensorBoard, self).__init__(self.log_dir_train, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.log_dir_val = f'{log_dir}/validation'

    def set_model(self, model):
        # Setup writer
        self._writer = {}
        self._writer['train'] = tf.summary.FileWriter(self.log_dir_train)
        self._writer['val'] = tf.summary.FileWriter(self.log_dir_val)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        metrics = Metrics(
            self.model,
            self.training_data,
            self.validation_data,
            self._writer,
            epoch)
        metrics.write_metrics()

        # Discard keras default logs
        logs = {}

        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        for name in ['val', 'train']:
            self._writer[name].close()


class Metrics:

    def __init__(self, model, train_data, validation_data, writer, epoch):
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
        self.writer = writer
        for name, data in [('val', validation_data), ('train', train_data)]:
            loss = model.evaluate(data[0], data[1])
            self.write(loss, 'loss_end_epoch', name)

    def write(self, value, plot_name, name):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = plot_name
        self.writer[name].add_summary(summary, self.epoch)

    def write_metrics(self):
    
        def calc_write(metric_func, target, predict):
            for name in ['val', 'train']:
                plot_name = metric_func.__name__
                metric_value = metric_func(target[name], predict[name])
                self.write(metric_value, plot_name, name)

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

        for name in ['val', 'train']:
            self.writer[name].flush()

    def precision_score(self, target_label, predict_label):
        return metrics.precision_score(target_label, predict_label, average='micro')

    def recall_score(self, target_label, predict_label):
        return metrics.recall_score(target_label, predict_label, average='micro')

    def f1_score(self, target_label, predict_label):
        return metrics.f1_score(target_label, predict_label, average='micro')

    def roc_auc_score(self, target_label, predict_label):
        return metrics.roc_auc_score(target_label, predict_label, average='micro')
