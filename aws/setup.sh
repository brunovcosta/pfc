#!/bin/sh
{
	# download code
	echo "git clone"
	git clone https://github.com/brunovcosta/PFC
	echo "pip install"
	pip install -r PFC/requirements.txt
	echo "download nltk datasets"
	python PFC/nltk_download.py

	# download glove
	echo "download glove_50.zip"
	echo "    download"
	wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s50.zip -O glove_50.zip
	echo "    unzip"
	unzip glove_50.zip
	mv glove_s50.txt PFC/dataset/glove

	echo "download rota_dos_concursos.csv"
	wget https://s3.us-east-2.amazonaws.com/pfc-ime/rota_dos_concursos.csv -O PFC/dataset/rota_dos_concursos.csv
	# run tests
	cd PFC
	echo "run tests"
	#python test/testing_bag_of_words.py &
	#python test/testing_simple_avg.py &
	#python test/testing_cnn.py &
	python test/testing_sepcnn.py#&
	#python test/testing_rnn.py

	#wait

	echo "upload logs"
	aws s3 sync PFC/logs s3://pfc-ime/logs
} >> /home/ubuntu/log_stdout 2>>/home/ubuntu/log_stderr
