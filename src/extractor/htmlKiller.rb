
while true
	for i in 0..999
		files = `cd ../../dataset/rawData/#{i} && ls -1`
		for file in files.split("\n")
			if file.end_with?(".html")
				newFile = file.split('.').first + ".json"
				puts newFile
				`ruby extractor.rb ../../dataset/rawData/"#{i}/#{file}" ../../dataset/rawData/"#{i}/#{newFile}"`
				`rm ../../dataset/rawData/"#{i}/#{file}"`
			end
		end
	end
	`sleep 30`
end