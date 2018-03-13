for i in 0..999
	files = `cd rawData/#{i} && ls -1`
	for file in files.split("\n")
		extractedText = `xmllint --html --format --xpath '//div[@class="page-header"]/h1/text()' rawData/#{i}/#{file} 2> /dev/null`
		if extractedText == "OOOPS!"
			`rm rawData/#{i}/#{file}`
		end
	end
end
