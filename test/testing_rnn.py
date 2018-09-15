import os
import sys

module_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(module_root))

from src.models.word_embedding.embedding_layer import RNN, StackedRNN, BidirectionalGRUConv, ConvLSTM
from src.datasetAPI import RotaDosConcursos

print("loading dataset...")
rota_dos_concursos = RotaDosConcursos(
    frac=1,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=10000)

model = RNN(
    rota_dos_concursos,
    n_features_per_word=100,
    hyperparameters_file="default")

model.summary(save_summary=True)

model.fit(
    save_metrics=True,
    save_checkpoints=True)

model.save_plots()
model.inspect_mispredictions('val', 10)
