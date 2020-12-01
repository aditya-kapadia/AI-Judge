# -*- coding: utf-8 -*-
from string import punctuation
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from array import array
import numpy as np
from gensim.models import Word2Vec

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


'''
The word2vec algorithm processes documents sentence by sentence. 
This means we will preserve the sentence-based structure during cleaning.



'''

# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines

'''
we adapt the process_docs() function to load and clean all 
of the documents in a folder and return a list of all document lines.

'''

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('caseT'):
			continue
		if not is_trian and not filename.startswith('caseT'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines

'''
 load all of the training data and convert it into a long list of ‘sentences’
 (lists of tokens) ready for fitting the word2vec model.

'''
# load training data
positive_lines = process_docs('dataset/affirmed', vocab, True)
negative_lines = process_docs('dataset/reversed', vocab, True)
sentences = negative_lines + positive_lines
print('Total training sentences: %d' % len(sentences))


'''
The word2vec algorithm processes documents sentence by sentence. 
This means we will preserve the sentence-based structure during cleaning.

The model is fit when constructing the class. We pass in the list of clean sentences
from the training data, then specify the size of the embedding vector space (we use 100 again),
the number of neighboring words to look at when learning how to embed each word in the training
sentences (we use 5 neighbors), the number of threads to use when fitting the model (we use 8, but change 
this if you have more or less CPU cores), and the minimum occurrence count for words to consider in the vocabulary
(we set this to 1 as we have already prepared the vocabulary).

'''

# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
'''
Vocabulary size should be same as that of vocab created  in Vectorizer

'''
print('Vocabulary size: %d' % len(words))

'''
Finally, we save the learned embedding vectors to file using the save_word2vec_format() on the model’s ‘wv‘ (word vector) attribute. 
The embedding is saved in ASCII format with one word and vector per line.

'''

# save model in ASCII (word2vec) format
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

'''
At first we created a random Vectors from Vectorizer class and then we trained it on Embedding Layer.Due to these 
Embedding layer takes more time.
To overcome these what we can do is Before fedding it to embedding layer we train the EMBEDDINGS on word2Vect class which
will give Meaningful vectors to each and every word Before going into Embedding layer . So it becomes easy for Embedding layer to learn

'''






