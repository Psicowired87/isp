import pandas as p

paths = ['test.csv', 'train.csv']
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])
print t #display the data

# print t.tweet[1] prints tweet1

import numpy
from sklearn.feature_extraction.text import TfidfVectorizer



