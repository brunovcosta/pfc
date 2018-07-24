import os
import sys

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.bag_of_words.classifiers import NB


model = NB(
    random_state=1,
    frac=1,
    dict_name="default.json",
    min_number_per_label=10000)
model.fit(save_metrics=True)

model.save_plots()

model.inspect_mispredictions('test', 3)