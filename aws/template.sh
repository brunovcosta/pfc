#!/bin/sh
{
	# aws
	sudo apt-get install unzip -y -y
	# install python3 env
	sudo add-apt-repository ppa:jonathonf/python-3.6
	sudo apt-get update -y
	sudo apt-get install python3.6 -y
	sudo apt-get install python3.6-tk -y
	sudo apt-get install python3.6-distutils -y
	wget https://bootstrap.pypa.io/get-pip.py
	source activate tensorflow_p36
	sudo python3.6 get-pip.py
	# download code
	echo "git clone"
	git clone https://github.com/brunovcosta/PFC
	echo "pip install"
	pip3.6 install -r PFC/requirements.txt
	echo "download nltk datasets"
	python3.6 PFC/nltk_download.py

	echo "download rota_dos_concursos.csv"
	wget https://s3.us-east-2.amazonaws.com/pfc-ime/rota_dos_concursos.csv -O PFC/dataset/rota_dos_concursos.csv

	echo "download glove"
	echo "download" ${n_features_per_word}
	wget http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s${n_features_per_word}.zip -O glove_50.zip
	unzip glove_${n_features_per_word}.zip
	mv glove_s${n_features_per_word}.txt PFC/dataset/glove

	echo "test all" ${n_features_per_word}
	python3.6 PFC/test/testing_${model}.py ${n_features_per_word}

	echo "upload logs"
	export AWS_ACCESS_KEY_ID=AKIAIK3CKW6PCAL2RUXA
	export AWS_SECRET_ACCESS_KEY=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
	export AWS_DEFAULT_REGION=us-east-2
	aws s3 sync PFC/logs s3://pfc-ime/logs
} >> /home/ubuntu/log_stdout 2>>/home/ubuntu/log_stderr
