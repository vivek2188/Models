from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()	# Porter Algorithm used for stemming

words = ['walking','walked','cunning','riding']
text = "Hello Dr. Eric, how are you doing today? Weather seems to be good. Actually, I am going outside to buy groceries. Would you like to come?"

print("Stemmed words are: ")
for word in word_tokenize(text):
	if word in [',','.','?','!']:
		continue
	print(ps.stem(word),end=",")
