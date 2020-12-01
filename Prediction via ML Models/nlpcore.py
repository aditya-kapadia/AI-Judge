# -*- coding: utf-8 -*-
#1.Importing
'''Import nltk and download all its classes and Libraries'''
import nltk
#nltk.download()

#2.Tokenization
''' We will divide the whole paragraph into different sentences and it will return list of all these sentences'''
paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""
               
#Tokenizing Sentence
sentences = nltk.sent_tokenize(paragraph)
#Tokenizing Word
word_token = nltk.word_tokenize(paragraph)


#3.Stemming and Lemmatization
'''
Stemming -> Will only keep root of word -> In review Loved will be replaced by root Love
We apply stemming to single word not on whole list

'''
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
sentences_stemmed=[]
for i in range(len(sentences)):
    word_token = nltk.word_tokenize(sentences[i])
    word_stem = [ps.stem(word) for word in word_token]
    #sentences[i] = ' '.join(word_stem)
    sentences_stemmed.append(' '.join(word_stem))

'''
Sometimes Stemming give root of words that have no meaning
i-e Final , Finally will be given root as 'Fina' which has no meaning
So to overcome these problem Lemmatization is used
Lemmatization first do stemming and then see whether the root has meaning if not then it finds
 
'''
from nltk.stem import WordNetLemmatizer
wl=WordNetLemmatizer()
sentences_lemmatized=[]
for i in range(len(sentences)):
    word_token=nltk.word_tokenize(sentences[i])
    word_lem=[wl.lemmatize(word)for word in word_token]
    sentences_lemmatized.append(' '.join(word_lem))

#4.Stopwords -> Remove non relvant words
nltk.download('stopwords')
from nltk.corpus import stopwords
sentences_stopword = []
for i in  range(len(sentences)):
    word_token=nltk.word_tokenize(sentences[i])
    word_stop=[word for word in word_token if not word in set(stopwords.words('english'))]
    sentences_stopword.append(' '.join(word_stop))


#Part of speech tagging
words = nltk.word_tokenize(paragraph)
words_tagged = nltk.pos_tag(words)
words_customized=[]
words_pos_para=[]
for i in words_tagged:
    words_customized.append(i[0]+"_"+i[1])
    
words_pos_para=' '.join(words_customized)


#Named Entity Recognization
''' It needs POS_TAGGING before Enitity Recognization'''
words = nltk.word_tokenize(paragraph)
words_tag = nltk.pos_tag(words)
named_entity= nltk.ne_chunk(words_tag)
named_entity.draw()

#Bag of word model
'''
We will take all different words from 1000 observation of reviews and will create 1 Column for each word
We will put these columns in a table where rows will be our all 1000 observations
Each cell represent no of times that word(Column) appeared in review
These whole table is knows as SPARSE MATRIX (Matrix contains lots of zero)
Now each Column in these table will acts as Independent Variables 
So thats why we need to do Cleaning process to reduce Independent varaiables
Bag of word Model is created through process of Tokenization(Create column of each word)

'''
import re
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
sentences = nltk.sent_tokenize(paragraph)
corpus=[]
for i in range(len(sentences)):
    sentences[i]= sentences[i].lower()
    sentences[i] = re.sub('[^a-zA-Z]',' ',sentences[i])
    sentences[i]=re.sub(r'\s+',' ',sentences[i])
    word_token = nltk.word_tokenize(sentences[i])
    wl=WordNetLemmatizer()
    sentences[i] = [wl.lemmatize(word)for word in word_token if not word in set(stopwords.words('english'))]
    corpus.append(' '.join(sentences[i]))
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100)
X=cv.fit_transform(corpus).toarray() 

'''
Bag of word model from Scratch

word2count = {}
for data in corpus:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
import heapq
import numpy as np
# Selecting best 100 features
freq_words = heapq.nlargest(100,word2count,key=word2count.get)

# Converting sentences to vectors
X_scratch = []
for data in corpus:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X_scratch.append(vector)
        
X_scratch= np.asarray(X_scratch)

'''

#TF-IDF model
'''
Problem in Bag of word model 
->All words have same importance
->No semantic information is preserved
Ex-> She is beautiful girl 
#Bag of word model -> All words have same importance
#TF-IDF model      -> Beautiful is given more importance than 'she is girl'

TF   ->Term frequency             -> No of occurence of word in document/No of words in that document
IDF  ->Inverse Document Frequency -> log(no of document/no of documents containing that word)
TF-IDF = TF*IDF

'''
#Creating Histogram
import heapq
import numpy as np
word2count = {}
for data in corpus:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
# Selecting best 100 features
freq_words = heapq.nlargest(100,word2count,key=word2count.get)


#IDF Matrix
word_idfs = {}
for word in freq_words:
     doc_count=0 #No of documents conatins that word
     for data in corpus:
         if word in nltk.word_tokenize(data):
             doc_count +=1 
     word_idfs[word] = np.log((len(corpus)/doc_count)+1)#+1 is bias
     
     
#TF Matrix
      
tf_matrix={}
for word in freq_words:
    doc_tf=[]
    for data in corpus:
        frequency=0
        for w in nltk.word_tokenize(data):
            if w==word:
                frequency +=1
        tf_word = frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word]= doc_tf
        
        
# Creating the Tf-Idf Model
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)  

# Finishing the Tf-Tdf model
X = np.asarray(tfidf_matrix)

X = np.transpose(X)     
        
    