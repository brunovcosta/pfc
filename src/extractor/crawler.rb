require 'nokogiri'
require 'json'

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
	fileCounter = 0
	nullCounter = 0

	threadNumber.step(maxId-1, numberOfThreads) do |fileNumber|
		htmlFile = "#{fileNumber}.html"
		puts "File " + htmlFile
		linkTest = linkPrefix + fileNumber.to_s

		`curl -sLg '#{linkTest}' > #{fileNumber%1000}/#{htmlFile}`
		extractedText = `xmllint --html --format --xpath '//div[@class="page-header"]/h1/text()' #{fileNumber%1000}/#{htmlFile} 2> /dev/null`
		
		if extractedText == "OOOPS!"
			puts "File #{htmlFile} was null"

			nullCounter = nullCounter + 1
		else
			fileCounter = fileCounter + 1
			jsonFile = "#{fileNumber}.json"
			extractor("#{fileNumber%1000}/#{htmlFile}", "#{fileNumber%1000}/#{jsonFile}")
		end
		
		`rm #{fileNumber%1000}/"#{htmlFile}"`

		puts "Thread: #{threadNumber}; File Count: #{fileCounter}; Null Count: #{nullCounter} (Total: #{fileCounter + nullCounter})"
	end
end

def extractor(input_path, output_path)
	input_file = open input_path
	output_file = File.new output_path, "w"

	html = Nokogiri::HTML input_file.read

	output_data = {
		id: (input_path.scan /\d+/).last,
		text: (html.css ".panel-questao .panel-heading").text.gsub(/\n/,'').squeeze(" "),
		subject_path: (html.css ".materia b").map{|b| b.text.gsub(/\n/,'').squeeze(" ")},
		alternatives: (html.css ".panel-questao .panel-body ul li").map{|li|li.text.gsub(/\n/,'').squeeze(" ").gsub(/^ ?[A-Z].\ /,'')},
		image_count: (html.css ".panel-questao .panel-heading img").count,
		concurso: (html.xpath "//div/p/a[contains(@href,'/concurso/')]").text.gsub(/\n/,'').squeeze(" "),
		prova: (html.xpath "//div/p/a[contains(@href,'/prova/')]").text.gsub(/\n/,'').squeeze(" "),
		banca: (html.xpath "//div/p/a[not(contains(@href,'/prova/') or contains(@href,'/concurso/'))]").text.gsub(/\n/,'').squeeze(" "),
		nivel: (html.xpath "//p[contains(text(),'Nível: ')]").text.gsub(/\n/,'').squeeze(" ").split(': ')[1],
		answer: ("ABCDE".index (html.css '.resposta-correta-escondida span').text.gsub(/\n|\ /,'')[0])
	}

	output_file.write output_data.to_json
end

threads = []

for threadNumber in 0..(numberOfThreads-1)
	puts "starting thread #{threadNumber}"
	threads.push(Thread.new(threadNumber){|threadNumber| crawler(maxId, numberOfThreads, threadNumber, linkPrefix)})
end

for thread in threads
	thread.join
end