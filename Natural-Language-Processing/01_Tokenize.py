#import nltk
#nltk.download()

from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello Dr. Eric, how are you doing today? Weather seems to be good. Actually, I am going outside to buy groceries. Would you like to come?"

# Sentence Segmentation/ Tokenization
print("Sentences are: ")
for index, sentence in enumerate(sent_tokenize(text)):
	print("{0}. {1}".format(index+1,sentence))

# Word Segmenation/ Tokenization
print("Words are: ",end="")
for word in word_tokenize(text):
	print(word,end=", ")
