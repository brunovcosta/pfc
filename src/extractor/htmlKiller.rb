
while true
	for i in 0..999
		files = `cd rawData/#{i} && ls -1`
		for file in files.split("\n")
			if file.end_with?(".html")
				newFile = file.split('.').first + ".json"
				puts newFile
				`ruby extractor.rb rawData/"#{i}/#{file}" rawData/"#{i}/#{newFile}"`
				`rm rawData/"#{i}/#{file}"`
			end
		end
	end
	`sleep 30`
end