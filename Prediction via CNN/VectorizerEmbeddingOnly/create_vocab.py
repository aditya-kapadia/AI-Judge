# -*- coding: utf-8 -*-
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r',encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

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

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('caseT'):
			continue
		if not is_trian and not filename.startswith('caseT'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('dataset/reversed', vocab, True)
process_docs('dataset/affirmed', vocab, True)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
#print(vocab.most_common(5000))


#Filterig out reviews that have not occured more than 2 times
# keep tokens with a min occurrence

min_occurane = 1
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
most_comman = (vocab.most_common(6575))


#Save tokens as Each word per line in vocab.txt
# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt') 






