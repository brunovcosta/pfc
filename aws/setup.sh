#!/bin/sh
{
	# download code
	echo "git clone"
	git clone https://github.com/brunovcosta/PFC
	echo "pip install"
	pip3 install -r PFC/requirements.txt
	echo "download nltk datasets"
	python3 PFC/nltk_download.py

	echo "download rota_dos_concursos.csv"
	wget https://s3.us-east-2.amazonaws.com/pfc-ime/rota_dos_concursos.csv -O PFC/dataset/rota_dos_concursos.csv

	echo "download glove"
	for n_features_per_word in 50 100 300 600 1000
	do
		echo "download" ${n_features_per_word}
		wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s${n_features_per_word}.zip -O glove_50.zip
		unzip glove_${n_features_per_word}.zip
		mv glove_s${n_features_per_word}.txt PFC/dataset/glove

		echo "test all" ${n_features_per_word}
		python3 PFC/test/all.py ${n_features_per_word}
	done

	echo "upload logs"
	aws s3 sync PFC/logs s3://pfc-ime/logs
} >> /home/ubuntu/log_stdout 2>>/home/ubuntu/log_stderr
