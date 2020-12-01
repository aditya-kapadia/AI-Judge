from flask import Flask,render_template,url_for,request
'''import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    dataset = pd.read_csv('Datasetaffirmed.csv')
    
    from nltk.corpus import stopwords
    #nltk.download('stopwords')
    #nltk.download('punkt')
    #nltk.download('wordnet')
	
    from nltk.stem import WordNetLemmatizer
    corpus = []
    
    for i in range(0,12):
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
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)
    
    from sklearn.svm import SVC
    classifier = SVC(C=1, kernel = 'rbf',gamma = 0.5 , random_state = 0)
    classifier.fit(X_train, y_train)
    '''
    y_pred = classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    '''
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        review = re.sub('[^a-zA-Z]',' ',str(data))

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
        review = [review]
        vect = cv.transform(review).toarray()
        vect = sc.transform(vect)
        
        #vect = cv.transform(data).toarray()
        
        
        my_prediction = classifier.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)