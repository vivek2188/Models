from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

words = ['walking','walked','cunning','riding']
text = "Hello Dr. Eric, how are you doing today? Weather seems to be good. Actually, I am going outside to buy groceries. Would you like to come?"

print("Lemmatizing: ",end="")
for word in word_tokenize(text):
	if word in [',','.','?','!']:
		continue
	print(lemmatizer.lemmatize(word),end=",")
