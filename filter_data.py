from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

source_file_name = "1000.txt"
write_file_name = "result.txt"

analyzer = SentimentIntensityAnalyzer()
old_line = ""

text_file = open(write_file_name, "w")

with open(source_file_name) as file:
    for line in file:
        sentiment_result = analyzer.polarity_scores(old_line)
        sentiment_result2 = analyzer.polarity_scores(line)
        if (sentiment_result['compound'] < -0.4) and (sentiment_result2['compound'] > 0.4) and (len(old_line) < 100) and (len(line) < 100):
            text_file.write("--------------------\n")
            text_file.write(old_line)
            text_file.write(line)
            text_file.write("--------------------\n")

        old_line = line

text_file.close()