import pandas as p

paths = ['train.csv', 'test.csv']
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])
print t #display the data

# print t.tweet[1] prints tweet1

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

## How to get the data into the right form?

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
tfidf.fit(t['tweet'])
X = tfidf.transform(t['tweet'])
test = tfidf.transform(t2['tweet'])
y = np.array(t.ix[:,4:])

## How to use sklearn to run an algorithm on the data?
# from sklearn.PathToRightAlgorithm import RightAlgorithm
# clf = RightAlgorithm()
# clf.fit(X,y)
# test_prediction = clf.predict(test)

from sklearn import linear_model
clt = linear_model.LinearRegression()
clt.fit(X,y) # doesn't work
