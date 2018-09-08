#!/bin/sh
# download code
git clone https://github.com/brunovcosta/PFC
pip install -r PFC/requirements.txt
python PFC/nltk_download.py

# download glove
wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s50.zip -O glove_50.zip
wget https://s3.us-east-2.amazonaws.com/pfc-ime/rota_dos_concursos.csv -O PFC/dataset/rota_dos_concursos.csv
unzip glove_50.zip
mv glove_s50.txt PFC/dataset/glove

# run tests
cd PFC
python test/testing_bag_of_words.py
python test/testing_simple_avg.py
python test/testing_cnn.py
python test/testing_rnn.py

aws s3 sync PFC/logs s3://pfc-ime/logs
