# Natural Language Processing

dataset = pd.read_csv('Datasetaffirmed.csv')





import re
review = re.sub('[^a-zA-Z]',' ',dataset['Cases'][0])

review=review.lower()

import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

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


    
    
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
corpus = []
for i in range(0,43):
    review = re.sub('[^a-zA-Z]',' ',dataset['Cases'][i])
    review=review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    sentences = nltk.sent_tokenize(review)
    word_token = nltk.word_tokenize(review)
    wl=WordNetLemmatizer()
    sentences_lemmatized=[]
    word_lem=[wl.lemmatize(word)for word in word_token]
    sentences_lemmatized.append(' '.join(word_lem))
    review = ' '.join(sentences_lemmatized)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500) #Top 1000 frequent words 
X=cv.fit_transform(corpus).toarray()
#Include Dependent Variable of each observation in dataset
y=dataset.iloc[:,1].values


#knn


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

'''
Choose number for k of neighbors
Find k number of neigbors of new point through Euclidean distance (Minkowski)
Among these k number of neighbors identified . Count the no points in each category of K 
Assign the new data point to the category which is having highest no of count
'''

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  





#SVM

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state=0)
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)#0 correct



#kernel svm
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=1, kernel = 'rbf',gamma = 0.5 , random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)







'''  GRID SEARCH  on kernel Svm '''
#Improving model Performance -> GRID SEARCH (To find best model and best parameters)
from sklearn.model_selection import GridSearchCV
'''

Specifying Different Parameters of which we have to find optimal values ->DICTIONARY
If GRID SEARCH tells kernel = 'linear' -> Linear Model
kernel = 'rbf' -> Non Linear Model
Use different values of parameter C of SVC 
Higher the value of c more  it will prevent overfitting 
Dont inc more otherwise it will be underfitting


'''
parameters = [ {'C':[1,10,100,1000],'kernel' : ['linear']}, #Option 1 -> Linear Model
               {'C':[1,10,100,1000],'kernel' : ['rbf'] ,'gamma':[0.5,0.1,0.01,0.001,0.0001]} #option 2 -> Non linear model
                ]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring ='accuracy',
                           #cv -> k fold cross validation
                           cv=5,
                           #For large dataset n_jobs=-1
                           #Will get error if used Be careful
                          )
grid_seach = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_









#Random forest

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
score = classifier.score(X_test,y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)#2 INCORRECT PREDICITIONS












#Naive Bayes

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = 0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB();
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)#1 INCORRECT 




#Performance Evaluation of model via K FOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
#Will return k no accuracies which is computed through k no of combinations
'''
estimator -> Your model i-e classifier
X -> Training set (Matrix of features)
y -> Dependent variables of Training set
cv -> No of FOLDS you want your training set to be splitted
If Large no  of dataset then
n_jobs -> -1  [Will use all your CPU's in machine to run k fold faster]



'''
accuracies = cross_val_score(estimator = classifier,X=X_train,y=y_train,cv=10)

#Take MEAN [ Average ] of all above accuraccies which are in vector
finalAccuracy = accuracies.mean()
#Standard Deviation -> Average of differences between different accuracies 
accuracies.std()




#Precision

TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]


accuracy = (TN+TP)/(TN+FP+FN+TP) * 100

precision = (TP)/(FP+TP) * 100

recall = (TP)/(FN+TP) * 100


f1_score = 2 *(recall * precision )/(recall+precision) * 100



2/3*100

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


















