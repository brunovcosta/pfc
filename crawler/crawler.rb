#!/usr/bin/env ruby

linkPrefix = "http://rotadosconcursos.com.br/questoes-de-concursos/filtro_avancado?anos[]=2018"

for i in 1997..2017
	linkPrefix= "#{linkPrefix}&anos[]=#{i}"
end

linkPrefix += "&id="

`mkdir rawData`

for i in 0..999
	`cd rawData/
	mkdir #{i}`
end

for i in 0..2000000
	linkTest = "#{linkPrefix}#{i}"
	`curl -sLg "#{linkTest}" > "rawData/#{i%1000}/#{i}.html"`
end
