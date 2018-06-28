import os
import sys

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.word_embedding import RNN, SimpleAvg, CNN


model = SimpleAvg(
    n_features_per_word=50,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=1000,
    frac=0.1)
model.fit(save_metrics=True)
model.summary()
model.inspect_mispredictions('test', 3)
