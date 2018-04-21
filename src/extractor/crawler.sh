#!/bin/bash

linkPrefix="http://rotadosconcursos.com.br/questoes-de-concursos/filtro_avancado?anos[]=2018"

for (( i=2017; i>1997; i--))
	do
		linkPrefix=$linkPrefix"&anos[]="$i
	done
linkPrefix=$linkPrefix"&id="

echo "Creating folders"
cd ../../dataset/
mkdir rawData
cd rawData
for (( i=0; i<1000; i++))
	do
		mkdir $i
		if [ "$(((i+1)%100))" == "0" ]; then
			echo "$(((i+1)/10))%"
		fi 
	done

let "maxId=2000000"
let "nullCounter=0"
for (( i=0; i<$maxId; i++))
	do
		newFile=$i".html"
		echo "File "$newFile
		cd $((i%1000))
		linkTest=$linkPrefix$i
		curl -sLg $linkTest > $newFile
		extractedText="$(xmllint --html --format --xpath '//div[@class="page-header"]/h1/text()' $newFile 2> /dev/null)"
		if [ "$extractedText" == "OOOPS!" ]; then
			echo "File "$newFile" was null"
			let "nullCounter = $nullCounter + 1"
			echo "Current fraction of null pages is around $((100*nullCounter/(i+1))) %"
			rm $newFile
		fi
		cd ..
	done
