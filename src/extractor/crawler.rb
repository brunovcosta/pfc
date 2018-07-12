linkPrefix = "http://rotadosconcursos.com.br/questoes-de-concursos/filtro_avancado?anos[]=2018"

for year in 2017.downto(1998)
	linkPrefix = linkPrefix + "&anos[]=#{year}"
end
linkPrefix = linkPrefix + "&id="

puts "Creating folders"
Dir.chdir "../../dataset/"
`mkdir rawData`
Dir.chdir "rawData"

for i in 0..1000
	`mkdir #{i}`
	if (i+1)%100 == 0
		puts "#{(i+1)/10}%"
	end
end

maxId = 2000000
nullCounter = 0
for fileNumber in 0..maxId
	htmlFile = "#{fileNumber}.html"
	puts "File " + htmlFile
	Dir.chdir "#{fileNumber%1000}"
	linkTest = linkPrefix + fileNumber.to_s
	`curl -sLg '#{linkTest}' > #{htmlFile}`
	extractedText = `xmllint --html --format --xpath '//div[@class="page-header"]/h1/text()' #{htmlFile} 2> /dev/null`
	
	if extractedText == "OOOPS!"
		puts "File #{htmlFile} was null"
		nullCounter = nullCounter + 1
		puts "Current fraction of null pages is around #{100*nullCounter/(i+1)} %"
	else
		jsonFile = "#{fileNumber}.json"
		`ruby ../../../src/extractor/extractor.rb "#{htmlFile}" "#{jsonFile}"`
	end
	
	`rm #{htmlFile}`
	
	Dir.chdir ".."
end