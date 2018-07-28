import os
import sys

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.word_embedding import CNN


model = CNN(
    n_features_per_word=50,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=10000,
    frac=1)

model.summary()

model.fit(
    save_metrics=True,
    save_checkpoints=False)

model.save_plots()
model.inspect_mispredictions('test', 10)
