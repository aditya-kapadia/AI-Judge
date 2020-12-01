#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
#We will use tsv -> Tab seperated value
#Dataset contains 1000 reviews
#We will ignore double quote by quoting = 3 
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#For any new Review our model will predict wether the review is postivie or negative

#Cleaning the Text
'''
Because Bag of Word model(Tokenization Process) only consist of getting only relevant words
where we will get rid of non relevant words such as 'the,on,as,a.....'
We will also get rid of punctuations
Stemming -> We will find different versions of same word which have same meaning
            Example -> Loved Love Loving etc
            Only keeping root of different version of words
            
We also get rid of Capital Letters ->All text will be Lower case

We will do all above steps to first review then will take for loop for next review onwards

'''
#re library for cleaning all texts
import re
'''

step1-> will only keep letters-> Remove number,punctuations(?,!,",...) etc
[^a-zA-Z] represents we dont want to remove' 
2nd param -> If ab 20 cd -> Then it would be abcd . we know ab and cd are different words to avoid it we use ' '
 
'''
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])#For First review

#Step2 -> Review in Lower case Letter
review=review.lower()

#Step3-> Removing non relevant words (Words that dont help ML model to predict)
import nltk 
#Tools that will help us to remove non relevant words 
nltk.download('stopwords')#List of words : Contains list of words that are not relvant 
from nltk.corpus import stopwords
'''
Now we will go through each word of review and see whether it is present in stopwords list if yes remove it from the review

'''
review = review.split()
review = [word for word in review if not word in set(stopwords.words('english'))]#if word is not in stopwords we will keep it in review set() helps to search faster

#step4->Stemming -> Will only keep root of word -> In review Loved will be replaced by root Love
#We apply stemming to single word not on whole list
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#Step5 Create a String of Cleaned review
review=' '.join(review) #' '. represents each word will be seperated by space


#step6 -> Do above 5 steps for all review in dataset
#Corpus means collection of text of same type
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    #steming
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
    
#Step 7 Bag of word 
'''
We will take all different words from 1000 observation of reviews and will create 1 Column for each word
We will put these columns in a table where rows will be our all 1000 observations
Each cell represent no of times that word(Column) appeared in review
These whole table is knows as SPARSE MATRIX (Matrix contains lots of zero)
Now each Column in these table will acts as Independent Variables 
So thats why we need to do Cleaning process to reduce Independent varaiables
Bag of word Model is created through process of Tokenization(Create column of each word)

'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()#When we use stopwords parameter It will automatically remove all stopwords from corpus
#fit_transform will create a sparse matrix
X_without_filtering=cv.fit_transform(corpus).toarray() #toarray ->To make whole table as Matrix

#To reduce the sparsity(No of zeros) in matrix we will use max-features parameter
'''
max_features parameters helps to remove non relevant words that appeared in only one or two review
Value of max_features represent Top frequent words in Corpus

'''
cv=CountVectorizer(max_features=1500) #Top 1000 frequent words 
X=cv.fit_transform(corpus).toarray()
#Include Dependent Variable of each observation in dataset
y=dataset.iloc[:,1].values

#Using Classification model for predictions
'''
Our model will find the correlations between independent and dependent variables and will make a predictions
Most common models used for Nlp is Decision tree and Naive Bayes and Random Forest
In these part we will use Naive Bayes
'''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#We dont need Feature scaling

# Fitting classifier to the Training set
# Create your classifier here
'''
It is probabilistic model where Bayes Theorem is used
p(a|b)=p(b|a)*p(a)/p(b)
So if there are two classes in data set 
Bayes theorem is applied to both of the classes
And whose probability is High that new data point will be under that category
'''

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB();
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#55correct predictions of Negative reviews
#91correct predictions of positive reviews
#12incorrect predictions of Negative review
#42incorrect predictions of Positive reviews 
#Total 54 Incorrect predictions
#Accuracy -> 146/200 ->73%










