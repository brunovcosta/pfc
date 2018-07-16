import tensorflow as tf
import numpy as np
import sklearn


class Metrics:

    def __init__(self, model, train_data, validation_data):
        self.predict_probability = {}
        self.predict_label = {}
        self.target_one_hot = {}
        self.target_label = {}
        raise NotImplementedError

    def accuracy_score(self, subset):
        return sklearn.metrics.accuracy_score(
            self.target_label[subset],
            self.predict_label[subset])

    def precision_score(self, subset):
        return sklearn.metrics.precision_score(
            self.target_label[subset],
            self.predict_label[subset],
            average='weighted')

    def recall_score(self, subset):
        return sklearn.metrics.recall_score(
            self.target_label[subset],
            self.predict_label[subset],
            average='weighted')

    def f1_score(self, subset):
        return sklearn.metrics.f1_score(
            self.target_label[subset],
            self.predict_label[subset],
            average='weighted')

    def roc_auc_score(self, subset):
        return sklearn.metrics.roc_auc_score(
            self.target_one_hot[subset],
            self.predict_probability[subset],
            average='weighted')


class MetricsBagOfWords(Metrics): #TODO

    def __init__(self, model, train_data, validation_data):
        self.predict_probability = {}
        self.predict_label = {}
        self.target_one_hot = {}
        self.target_label = {}

        for subset, data in [('val', validation_data), ('train', train_data)]:
            self.predict_probability[subset] = np.asarray(model.predict_proba(data[0]))
            self.predict_label[subset] = np.argmax(self.predict_probability[subset], axis=1)
            self.target_one_hot[subset] = np.asarray(data[1])
            self.target_label[subset] = np.argmax(self.target_one_hot[subset], axis=1)

    def save_results(self):
        pass

        def calc_save(metric_func):
            pass
            for subset in ['val', 'train']:
                plot_name = metric_func.__name__
                metric_value = metric_func(subset)

        metric_functions = [
            self.accuracy_score,
            self.precision_score,
            self.recall_score,
            self.f1_score,
            self.roc_auc_score
        ]

        for metric_func in metric_functions:
            calc_save(metric_func)


class MetricsTensorboard(Metrics):

    def __init__(self, model, train_data, validation_data):
        self.predict_probability = {}
        self.predict_label = {}
        self.target_one_hot = {}
        self.target_label = {}
        self.loss = {}

        for subset, data in [('val', validation_data), ('train', train_data)]:
            self.predict_probability[subset] = np.asarray(model.predict(data[0]))
            self.predict_label[subset] = np.argmax(self.predict_probability[subset], axis=1)
            self.target_one_hot[subset] = np.asarray(data[1])
            self.target_label[subset] = np.argmax(self.target_one_hot[subset], axis=1)
            self.loss[subset] = model.evaluate(data[0], data[1])

    def write_metrics_to_tensorboard(self, writer, epoch):

        def write(value, plot_name, subset):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = plot_name
            writer[subset].add_summary(summary, epoch)

        for subset in ['val', 'train']:
            write(self.loss[subset], 'loss_end_epoch', subset)

        def calc_write(metric_func):
            for subset in ['val', 'train']:
                plot_name = metric_func.__name__
                metric_value = metric_func(subset)
                write(metric_value, plot_name, subset)

        metric_functions = [
            self.accuracy_score,
            self.precision_score,
            self.recall_score,
            self.f1_score,
            self.roc_auc_score
        ]

        for metric_func in metric_functions:
            calc_write(metric_func)

        for subset in ['val', 'train']:
            writer[subset].flush()
