launch = File.new("launch.sh","a")
launch.write "export AWS_PROFILE=pfc"
for n in [50,100,300,600,1000] do
	path_start = "../test/testing_"
	path_end = ".py"
	for file in Dir["#{path_start}*#{path_end}"] do
		model = file.gsub(path_start,"").gsub(path_end,"")
		rendered = open("./template.sh").read.gsub("${n_features_per_word}",n.to_s).gsub("${model}",model)
		File.open("setup_#{n}_#{model}.sh","w").write rendered
		launch.write "\naws ec2 run-instances --cli-input-json file://instance.json --user-data file://setup_#{n}_#{model}.sh"
	end
end
