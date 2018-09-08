import os
import sys

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.word_embedding import SimpleAvg


model = SimpleAvg(
    n_features_per_word=50,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=10000,
    frac=1)

model.summary(save_summary=True)

model.fit(
    save_metrics=True,
    save_checkpoints=True)

# model.load_model("run-20180718142233-SimpleAvg_50")

model.save_plots()
model.inspect_mispredictions('val', 10)
