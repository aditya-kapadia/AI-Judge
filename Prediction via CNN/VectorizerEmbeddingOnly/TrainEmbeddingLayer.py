
# -*- coding: utf-8 -*-
from string import punctuation
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from array import array
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# load doc into memory
def load_doc(filename):
	
# open the file as read only
	file = open(filename, 'r',encoding="ISO-8859-1")
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

Next, we need to load all of the training data movie reviews. For that we can adapt the process_docs() from the previous section to load the documents, 
clean them, and return them as a list of strings, with one document per string. We want each document to be a string for easy encoding as a sequence of integers later.

'''
# turn a doc into clean tokens
def clean_doc(doc, vocab):
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
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens
    '''
    	# split into tokens by white space
        ens = doc.split()
        emove punctuation from each token
        le = str.maketrans('', '', punctuation)
        ens = [w.translate(table) for w in tokens]
        ilter out tokens not in vocab
        ens = [w for w in tokens if w in vocab]
        ens = ' '.join(tokens)
        urn tokens
    '''

# load all docs in a directory

def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('caseT'):
			continue
		if not is_trian and not filename.startswith('caseT'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# load all training reviews
positive_docs = process_docs('dataset/affirmed', vocab, True)
negative_docs = process_docs('dataset/reversed', vocab, True)
train_docs = negative_docs + positive_docs


'''
We can encode the training documents as sequences of integers using the Tokenizer class in the Keras API.
it develops a vocabulary of all tokens in the training dataset and develops a consistent mapping from words 
in the vocabulary to unique integers

'''
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

'''
Now that the mapping of words to integers has been prepared, we can use it to encode the reviews in the 
training dataset.

'''

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)

'''
All documents should have same length
we can find the longest review using the max() function on the training dataset and take its length
'''
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

'''
we can define the class labels for the training dataset, needed to fit the supervised neural network model to
predict the sentiment of reviews.

'''

ytrain = np.array([0 for _ in range(17)] + [1 for _ in range(17)])


'''

Need to use it on flask 

We can then encode and pad the test dataset, needed later to evaluate the model after we train it.

'''
# load all test reviews
positive_docs_t = process_docs('dataset/affirmed', vocab, False)
negative_docs_t = process_docs('dataset/reversed', vocab, False)
test_docs_t = negative_docs_t + positive_docs_t
# sequence encode
encoded_docs_t = tokenizer.texts_to_sequences(test_docs_t)
# pad sequences
Xtest = pad_sequences(encoded_docs_t, maxlen=max_length, padding='post')
# define test labels
ytest = np.array([0 for _ in range(2)] + [1 for _ in range(2)])


'''   NEURAL NETWORK   '''
'''
The model will use an Embedding layer as the first hidden layer. The Embedding requires the specification 
of the vocabulary size, the size of the real-valued vector space, and the maximum length of input documents.

'''

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

'''
We will use a 100-dimensional vector space, but you could try other values, such as 50 or 150.
The maximum document length was calculated above in the max_length variable used during padding

'''

#CNN

'''
The complete model definition is listed below including the Embedding layer.

We use a Convolutional Neural Network (CNN) as they have proven to be successful at document classification problems. 
A conservative CNN configuration is used with 32 filters (parallel fields for processing words) and a kernel size of 8
with a rectified linear (‘relu’) activation function. 
This is followed by a pooling layer that reduces the output of the convolutional layer by half.
Next, the 2D output from the CNN part of the model is flattened to one long 2D vector to represent the ‘features’ extracted by the CNN. The back-end of the model is a standard Multilayer Perceptron layers to interpret the CNN features. The output layer uses a sigmoid activation function to output a value between 0 and 1 for the negative and positive sentiment in the review.

'''

# define model
model = Sequential()
model.add(Embedding(vocab_size, 150, input_length=max_length))
model.add(Conv1D(filters=128, kernel_size=16, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(filters=128, kernel_size=16, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())


# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=100, verbose=2)

# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))


#Save the model to disk
filename = 'model.pkl'
pickle.dump(model,open(filename,'wb'))





#Loading the saved model
model = pickle.load(open(filename,'rb'))
loss,result = model.evaluate(Xtest,ytest,verbose=0)
print('Test Accuracy: %f' % (result*100))




#Evaluation of CNN Matrix

# predict probabilities for test set
yhat_probs = model.predict(Xtest, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(Xtest, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(ytest, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(ytest, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(ytest, yhat_classes)
print(matrix)
