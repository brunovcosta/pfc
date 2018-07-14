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
end

maxId = 2000000
$nullCounter = 0
numberOfThreads = ARGV[0].to_i || 4

puts "Number of threads: #{numberOfThreads}"

def crawler(maxId, numberOfThreads, threadNumber, linkPrefix)
	puts "we are in the thread #{threadNumber}"
	inferiorLimit = threadNumber*maxId/numberOfThreads
	superiorLimit = inferiorLimit + maxId/numberOfThreads - 1 
	threadNumber.step(maxId-1, numberOfThreads) do |fileNumber|
		htmlFile = "#{fileNumber}.html"
		puts "File " + htmlFile
		linkTest = linkPrefix + fileNumber.to_s

		`curl -sLg '#{linkTest}' > #{fileNumber%1000}/#{htmlFile}`
		extractedText = `xmllint --html --format --xpath '//div[@class="page-header"]/h1/text()' #{fileNumber%1000}/#{htmlFile} 2> /dev/null`
		
		if extractedText == "OOOPS!"
			puts "File #{htmlFile} was null"
			$nullCounter = $nullCounter + 1
			puts "Current fraction of null pages is around #{100*$nullCounter/(maxId)} %"
		else
			jsonFile = "#{fileNumber}.json"
			`ruby ../../src/extractor/extractor.rb #{fileNumber%1000}/"#{htmlFile}" #{fileNumber%1000}/"#{jsonFile}"`
		end
		
		`rm #{fileNumber%1000}/"#{htmlFile}"`
	end
end

threads = []

for threadNumber in 0..(numberOfThreads-1)
	puts "starting thread #{threadNumber}"
	threads.push(Thread.new(threadNumber){|threadNumber| crawler(maxId, numberOfThreads, threadNumber, linkPrefix)})
end

for thread in threads
	thread.join
end