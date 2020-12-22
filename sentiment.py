# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:34:17 2020

@author: Voilet Pince
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
import pickle
import warnings

warnings.filterwarnings('ignore')

#importing the datasets
ds = pd.read_csv("labeledTrainData.tsv", delimiter="\t")
train = ds.drop(['id'], axis=1)

X1 = train['review'].values
y = train['sentiment'].values

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words = 'english', ngram_range = (1,2), tokenizer = token.tokenize).fit(X1)
X = cv.transform(X1)

#splitting gthe dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= True, random_state=42)

# training our classifier ; train_data.target will be having numbers assigned for each category in train data
clf = MultinomialNB().fit(X_train, y_train)

pickle.dump(clf,open('NBmodel.pkl','wb'))
pickle.dump(cv,open('vector.pkl','wb'))

predicted = clf.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)
print(str('{:04.2f}'.format(accuracy_score * 100))+'%')