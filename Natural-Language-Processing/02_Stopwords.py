from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "I guess this could be the nice example for showing the stopwords removal technique."

stop_words = set(stopwords.words("english"))
#print(stop_words)

orig_sent = word_tokenize(text)
print("Given sentence: ", orig_sent)

filtered_sent = list()
for word in word_tokenize(text):
	if word in stop_words:	# Removing the stopwords
		continue
	filtered_sent.append(word)

print("Filtered sentence: ", filtered_sent)
