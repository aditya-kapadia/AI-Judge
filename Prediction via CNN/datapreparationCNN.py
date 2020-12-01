# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import string
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	#tokens = doc.split()
    
    review = re.sub('[^a-zA-Z]',' ',doc)

    review=review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    sentences = nltk.sent_tokenize(review)
    word_token = nltk.word_tokenize(review)

    from nltk.stem import WordNetLemmatizer
    wl=WordNetLemmatizer()
    sentences_lemmatized=[]
    for i in range(len(sentences)):
        word_token=nltk.word_tokenize(sentences[i])
        word_lem=[wl.lemmatize(word)for word in word_token]
        sentences_lemmatized.append(' '.join(word_lem))
        review = ' '.join(sentences_lemmatized)
    tokens=review.split()
    return tokens

# load the document
filename = 'dataset/affirmed/case1.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)