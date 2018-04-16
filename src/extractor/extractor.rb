require 'nokogiri'
require 'json'

input_path = ARGV[0]
output_path = ARGV[1]

input_file = open input_path
output_file = File.new output_path, "w"

html = Nokogiri::HTML input_file.read

output_data = {
	id: (html.css "h1.panel-title").text.gsub(/\n/,'').squeeze.scan(/\d+/)[0],
	text: (html.css ".panel-questao .panel-heading").text.gsub(/\n/,'').squeeze,
	subject_path: (html.css ".materia b").map{|b| b.text.gsub(/\n/,'').squeeze},
	alternatives: (html.css ".panel-questao .panel-body ul li").map{|li|li.text.gsub(/\n/,'').squeeze.gsub(/^ ?[A-Z].\ /,'')},
	image_count: (html.css ".panel-questao .panel-heading img").count,
	concurso: (html.xpath "//div/p/a[contains(@href,'/concurso/')]").text.gsub(/\n/,'').squeeze,
	prova: (html.xpath "//div/p/a[contains(@href,'/prova/')]").text.gsub(/\n/,'').squeeze,
	banca: (html.xpath "//div/p/a[not(contains(@href,'/prova/') or contains(@href,'/concurso/'))]").text.gsub(/\n/,'').squeeze,
	nivel: (html.xpath "//p[contains(text(),'Nível: ')]").text.gsub(/\n/,'').squeeze.split(': ')[1],
	answer: ("ABCDE".index (html.css '.resposta-correta-escondida span').text.gsub(/\n|\ /,'')[0])
}

output_file.write output_data.to_json
