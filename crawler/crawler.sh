#!/bin/bash

linkPrefix="http://rotadosconcursos.com.br/questoes-de-concursos/filtro_avancado?anos[]=2018"

for (( i=2017; i>1997; i--))
	do
		linkPrefix=$linkPrefix"&anos[]="$i
	done

linkPrefix=$linkPrefix"&id="
mkdir rawData

for (( i=0; i<1000000; i++))
	do
		linkTest=$linkPrefix$i
		curl -sLg $linkTest > "rawData/"$i".html"
	done
