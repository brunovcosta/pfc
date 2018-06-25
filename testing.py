from src.models.word_embedding.rnn import RNN

sa = RNN(
    n_features_per_word=50,
    random_state=1,
    dict_name="default.json",
    min_number_per_label=1000,
    frac=0.1)
sa.execute_model()
