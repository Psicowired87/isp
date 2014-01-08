from __future__ import print_function
import pandas as p
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix, hstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, scale
from nltk.corpus import stopwords

from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import *


from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import *
from sklearn.multiclass import *

import time
import emo
import re

import string
from textblob import TextBlob
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
from preprocessing import Preprocess

import multiprocessing
from multiprocessing import Pool

import csv
from collections import defaultdict
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import random

from sklearn.metrics import accuracy_score
from numpy import ma
from sklearn.cross_validation import ShuffleSplit

from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix, hstack
from sklearn.ensemble import AdaBoostRegressor




class Model:

    def __init__(self):
        self.clfs_file = 'my_clfs.p' # file to store clfs
        self.train_file = 'data/train.csv'
        self._clfs = None


    @property
    def clfs(self):
        if not self._clfs:
            self.loadclfs()
        return self._clfs


    def train(self):
        ###################
        #Train
        ###################
        train = run_load_data(self.train_file)  # load training data
        pr_train_tweets = run_preprocess(train) # preprocess training data
        clf_features, X_train = run_feature_extraction(train.ix[:,1].tolist(), pr_train_tweets) # extract features from training data

        y = np.array(train.ix[:,4:])
        clfs_model = train(X_train, y) # train model

        pickle.dump((clfs_model, clf_features), open(self.clfs_file, 'wb')) #serialize clfs
        self._clfs = (clfs_model, clf_features)


    def predict(self, tweets):

        ###################
        #Test
        ###################

        loaded_clfs_model = self.clfs[0] # clfs for trained model
        loaded_clf_features = self.clfs[1] # clf for feature extraction

        pr_test_tweets = run_preprocess(tweets) # preprocess given test tweet!!
        X_test = run_feature_extraction(tweets, pr_test_tweets, clf=loaded_clf_features) #extract features for given tweet

        test_prediction = run_predict(loaded_clfs_model, X_test) #predict labels for test tweet
        return test_prediction

    def loadclfs(self):
        try:
            self._clfs = pickle.load(open(self.clfs_file, 'rb')) #deserialize clfs
        except IOError as e:
            self.train()

class ModelException(Exception):
    pass



############# LOAD THE DATA #############
def read_tagged_tweets(fname):

	list_tagged_tweets = []

	with open(fname) as tsv:#'tagged_tweets.txt'
		aTagged_tweet = []
		for line in csv.reader(tsv, delimiter='\t', quoting=csv.QUOTE_NONE):
			if len(line) > 0:
				aTagged_tweet.append(tuple([line[0].lower(), line[1]]))
			else:
				list_tagged_tweets.append(aTagged_tweet)
				aTagged_tweet = []
	return list_tagged_tweets



def load_data(trainFile='data/train.csv', testFile='data/test.csv'):
	print('Loading data...', end=' ')
	timestamp1 = time.time()

	global t, t2
	t = p.read_csv(trainFile)
	t2 = p.read_csv(testFile)
	
	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))



########### PREPROCESSING ######################
def preprocess_data(method=1):
	print('Preprocessing...', end=' ')
	timestamp1 = time.time()
	

	pr = Preprocess(rm_punctuation=True, rm_special=True, special_words=['mention', 'rt', 'link'], split_compound=True, translate=False, correct=False, singularize=False, 
		lemmatize=False, abbs_file = 'tweeter_abbreviations.csv', expand_abbs=True, substitute_url=True, rm_digits=False, rm_repeated_chars=False, rm_single_chars=False)

	if method==1:
		tagged_train = read_tagged_tweets('tagged_tweets/tagged_train_tweets.txt')
		tagged_test = read_tagged_tweets('tagged_tweets/tagged_test_tweets.txt')
		pr_train_tweets = pr.runTaggedParallel(tagged_train)
		pr_test_tweets = pr.runTaggedParallel(tagged_test)
	else:
		train_tweets = t['tweet'].tolist()
		test_tweets = t2['tweet'].tolist()
		pr_train_tweets = pr.runParallel(train_tweets)
		pr_test_tweets = pr.runParallel(test_tweets)


	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	return (pr_train_tweets, pr_test_tweets)



def feature_extraction5(train_tweets, test_tweets, maxFeatures2=35000, ngrams2=(1,3), maxdf2=[1.0], mindf2=0.0, isbinary2=True, method2=1):

	Fs = len(maxdf2)*[None]
	for i, aMax in enumerate(maxdf2):
		X, y, test, fn = feature_extraction(train_tweets, test_tweets, maxFeatures=maxFeatures2, ngrams=ngrams2, maxdf=aMax, mindf=mindf2, isbinary=isbinary2, method=method2)
		Fs[i] = (X, y, test, fn)

	return dict(zip(maxdf2, Fs))



############# FEATURE EXTRACTION #############
def feature_extraction(train_tweets, test_tweets, maxFeatures=35000, ngrams=(1,3), maxdf=0.95, mindf=0.0, isbinary=True, method=1):
	#print('Feature extraction...', end=' ')
	#timestamp1 = time.time()


	#tfidf = CountVectorizer(decode_error=u'ignore', charset_error=None, stop_words='english', strip_accents='unicode', lowercase=True, ngram_range=(1, 3), analyzer=u'word', max_df=0.9, min_df=0.0001, max_features=16000, binary=True, preprocessor=pr)
	#tfidf = CountVectorizer(decode_error=u'ignore', charset_error=None, stop_words=stopwords.words('english'), strip_accents='unicode', lowercase=True, ngram_range=(1, 3), analyzer=u'word', max_df=0.9, min_df=0.0001, max_features=16000, binary=True, preprocessor=pr)
	#tfidf = CountVectorizer(decode_error=u'ignore', charset_error=None, strip_accents='unicode', lowercase=True, ngram_range=ngrams, analyzer=u'word', max_df=maxdf, min_df=mindf, max_features=maxFeatures, binary=isbinary)
	#tfidf = CountVectorizer(decode_error=u'ignore', charset_error=None, strip_accents='unicode', lowercase=True, ngram_range=(1, 3), analyzer=u'word', min_df=3, max_features=16000, binary=True)

	tfidf = None
	if method ==1:
		tfidf = CountVectorizer(decode_error=u'ignore', charset_error=None, strip_accents='unicode', lowercase=True, ngram_range=ngrams, analyzer=u'word', max_df=maxdf, min_df=mindf, max_features=maxFeatures, binary=isbinary)
	else:
		tfidf = TfidfVectorizer(decode_error=u'ignore', strip_accents='unicode', analyzer=u'word', lowercase=True, ngram_range=ngrams, max_df=maxdf, min_df=mindf, max_features=maxFeatures, binary=isbinary)

	tfidf.fit(train_tweets)
	X = tfidf.transform(train_tweets)
	y = np.array(t.ix[:,4:])
	test = tfidf.transform(test_tweets)

	l = emoticones_feature(t.ix[:,1].tolist())
	emo_feature = preprocess_emoticones_feature(["NA", "HAPPY", "SAD"], l)
	X = hstack([X, emo_feature])

	l = emoticones_feature(t2.ix[:,1].tolist())
	emo_feature = preprocess_emoticones_feature(["NA", "HAPPY", "SAD"], l)
	test = hstack([test, emo_feature])

	sent_feature = sentimentFeatures(t.ix[:,1].tolist(), type='both')	
	X = hstack([X, sent_feature])
	sent_feature = sentimentFeatures(t2.ix[:,1].tolist(), type='both')
	test = hstack([test, sent_feature])




	#keywords=["cloud", "cold", "dry", "hot", "humid", "hurricane", 
	#"ice", "rain","drizzle", "snow", "storm", "sun", 
	#"tornado", "wind","downpour","flood","freeze","hail","lightning",
	#"overcast","thunder","typhoon"]

	keywords=['drizzle', 'flood','cloud', 'swarm', 'fog', 'gloom', 'haze', 'mist', 'obfuscate', 
	'obscure', 'overshadow', 'shade', 'rack', 'haziness', 'puff', 
	'billow', 'frost', 'nebula', 'nebulosity', 'vapor', 'veil', 
	'overcast', 'pall', 'brume', 'cold', 'arctic', 'chill', 
	'chilly', 'freezing', 'frigid', 'frigidity', 'frosty', 
	'frozen', 'glacial', 'iciness', 'icy', 'shivery', 'wintry', 
	'crisp', 'gelid', 'polar', 'dry', 'arid', 'juiceless', 
	'barren', 'drained', 'droughty', 'bare', 'dusty', 'hot', 
	'warm', 'heated', 'randy', 'ardent', 'fervent', 'glowing', 
	'burning', 'humid', 'clammy', 'steamy', 'hurricane', 
	'typhoon', 'cyclone', 'ice', 'rain', 'shower', 
	'wet', 'downpour', 'snow', 'storm', 'tempest', 'squall', 
	'gale', 'sun', 'tornado', 'whirlwind', 'twister', 'blow', 
	'wind', 'breeze', 'draught', 'draft', 'mistral', 
	'gust', 'blast', 'flurry', 'whisk', 'whiff', 
	'flutter', 'wafting', 'sirocco', 'hail','lightning']

	X = hstack([X, new_features2(keywords,t.ix[:,1].tolist())])
	test = hstack([test, new_features2(keywords,t2.ix[:,1].tolist())])

	# append wind 
	aReWind = re.compile('(\d+\.?\d+)[ \t]*mph', re.IGNORECASE)
	X = hstack([X, new_features3(t.ix[:,1].tolist(), aReWind)])
	test = hstack([test, new_features3(t2.ix[:,1].tolist(), aReWind)])

	# append humidity 
	#aReHumidity = re.compile('hum\w*[ \t]*\:[ \t]*(\d+\.?\d+)[ \t]*(&#x25;|%)?|(\d+\.?\d+)[ \t]*(&#x25;|%)?[ \t]*hum\w*', re.IGNORECASE)
	#X = hstack([X, new_features3(t.ix[:,1].tolist(), aReHumidity)])
	#test = hstack([test, new_features3(t2.ix[:,1].tolist(), aReHumidity)])
	
	# append rain
	#aReRain = re.compile('rain[ \t]*\:?[ \t]*(\d+\.?\d+)', re.IGNORECASE)
	#X = hstack([X, new_features3(t.ix[:,1].tolist(), aReRain)])
	#test = hstack([test, new_features3(t2.ix[:,1].tolist(), aReRain)])

	#append state features to train and test
	#state_col = p.concat([t.ix[:,3], t2.ix[:,3]])
	#state_feature = preprocess_categorical_feature(state_col)
	#X = hstack([X, state_feature[0:X.shape[0]]])
	#test = hstack([test, state_feature[X.shape[0]:]])

	#timestamp2 = time.time()
	#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	fn = tfidf.get_feature_names()

	return (X, y, test, fn)


def my_score_func(X, y, aCenter=False):
	return f_regression(X, y, center=aCenter)


def feature_selection(X, y, perc=60):
	print('Feature selection...', end=' ')
	timestamp1 = time.time()

	selector = SelectPercentile(my_score_func, percentile=perc)
	selector.fit(X, y)
	
	idx_features = selector.get_support(indices=True)

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	return idx_features



############# TRAINING #############
def train(X, y, tfidf=1):
	print('Training...', end=' ')
	timestamp1 = time.time()

	best_alphasTFIDF = [7.5, 1.5, 2.5, 2, 2, 3, 2.5, 4.5, 2, 3, 2.5, 3, 2.5, 1.5, 5, 3, 1.5, 7.5, 2, 4.5, 4.5, 3, 2.5, 1.5]
	best_alphasCV = [100, 20, 50, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 20, 10, 50, 20, 100, 20, 50, 50, 50, 20, 20]
	
	
	clf = linear_model.Ridge(alpha = best_alphasTFIDF)
	if tfidf!=1:
		clf = linear_model.Ridge(alpha = best_alphasCV)
		
	clf.fit(X,y)

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	# Print train RMSE
	#train_prediction = clf.predict(X)

	#new_array = train_prediction

	#print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(new_array-y)**2)/ (X.shape[0]*24.0))))
	#print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/ (X.shape[0]*24.0))))
	
	return clf


def my_custom_loss_func(y, pred):
	return np.sqrt(np.sum(np.array(pred-y)**2)/ (y.shape[0]*24.0))

def train2(X, y):
	print('Training...', end=' ')
	timestamp1 = time.time()

	my_custom_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)
	
	tuned_parametersRidge = {'alpha': [5, 10, 15, 20, 25, 35, 40, 50, 100, 200]}
	tuned_parametersRidge = {'alpha': [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 7.5, 10]}
	#tuned_parametersRidge = {'alpha': [5, 10, 15, 20, 25, 35, 40, 50, 100, 150], 'fit_intercept':[True, False], 'normalize':[True, False], 'tol':[0.01, 0.001, 0.0001]}


	clfs=y.shape[1]*[None]
	for i in range(y.shape[1]):
		#clfs[i] = linear_model.Ridge(alpha = 0.05)
		#clfs[i] = GridSearchCV(linear_model.Ridge(), param_grid=tuned_parametersRidge, cv=5, scoring='mean_squared_error', n_jobs=6)
		clfs[i] = GridSearchCV(linear_model.Ridge(), param_grid=tuned_parametersRidge, cv=5, scoring=my_custom_scorer, n_jobs=6)
		#clfs[i] = linear_model.Lasso(alpha = 0.15)
		#clfs[i] = linear_model.LassoLars(alpha=.15) #requires dense matrix
		#clfs[i] = linear_model.ElasticNet(alpha=0.15, l1_ratio=0.7)
		#clfs[i] = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=6000) #requires dense matrix
		#clfs[i] = linear_model.BayesianRidge(compute_score=True) #requires dense matrix
		#clfs[i] = linear_model.ARDRegression(compute_score=True) #requires dense matrix
		#clfs[i] = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
		#clfs[i] = svm.SVR(kernel='linear', C=1e3)
		#clfs[i] = svm.SVR(kernel='poly', C=1e3, degree=2)
		clfs[i].fit(X,y[:,i])
		print("Grid Search: best score and params for label {0}:".format(i))
		print(clfs[i].best_score_)
		print(clfs[i].best_params_)
		print()

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	
	return clfs



def train3old(Xtfidf, Xcv, y):
	print('Training...', end=' ')
	timestamp1 = time.time()

	best_alphasTFIDF = [7.5, 1.5, 2.5, 2, 2, 3, 2.5, 4.5, 2, 3, 2.5, 3, 2.5, 1.5, 5, 3, 1.5, 7.5, 2, 4.5, 4.5, 3, 2.5, 1.5]
	best_alphasCV = [100, 20, 50, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 20, 10, 50, 20, 100, 20, 50, 50, 50, 20, 20]
	
#	test_prediction = np.zeros(shape=y.shape)
	clfs=y.shape[1]*[None]
	for i in range(y.shape[1]):

		clfs[i] = linear_model.Ridge(alpha = best_alphasTFIDF[i])
		#clfs[i] = AdaBoostRegressor(base_estimator=linear_model.Ridge(alpha = best_alphasTFIDF[i]), n_estimators=5, random_state=None)
		X = Xtfidf

#		if i>0:
#			X = hstack([Xtfidf, test_prediction[:,0:i]])
		if i in [10, 12, 15, 16, 18, 20, 21, 23]:
			clfs[i] = linear_model.Ridge(alpha = best_alphasCV[i])
			#clfs[i] = AdaBoostRegressor(base_estimator=linear_model.Ridge(alpha = best_alphasCV[i]), n_estimators=5, random_state=None)
			X=Xcv
#			if i>0:
#				X = hstack([Xcv, test_prediction[:,0:i]])

		#X = X.toarray()

		clfs[i].fit(X,y[:,i])
#		test_prediction[:,i] = clfs[i].predict(X)

		#del X

#		x = ma.array(test_prediction[:,i])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,i] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,i] = x1.filled(1)


	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	
	return clfs


def train5(dict_Xtfidf, dict_Xcv, y):
	print('Training...', end=' ')
	timestamp1 = time.time()

	mymin = dict(zip([3, 9, 14, 16, 18, 19, 22, 23], [0.4, 0.4, 0.375, 0.4, 0.4, 0.3, 0.375, 0.35]))

	best_alphasTFIDF = [7.5, 1.5, 2.5, 2, 2, 3, 2.5, 4.5, 2, 3, 2.5, 3, 2.5, 1.5, 5, 3, 1.5, 7.5, 2, 4.5, 4.5, 3, 2.5, 1.5]
	best_alphasCV = [100, 20, 50, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 20, 10, 50, 20, 100, 20, 50, 50, 50, 20, 20]
	
	
	clfs=y.shape[1]*[None]
	for i in range(y.shape[1]):

		
		clfs[i] = linear_model.Ridge(alpha = best_alphasTFIDF[i])
		X = dict_Xtfidf[0.45][0]
		if i in [3, 9, 14, 16, 18, 19, 22, 23]:
			X = dict_Xtfidf[mymin[i]][0]

		if i in [10, 12, 15, 16, 18, 20, 21, 23]:
			clfs[i] = linear_model.Ridge(alpha = best_alphasCV[i])
			X = dict_Xcv[0.45][0]
			if i in [3, 9, 14, 16, 18, 19, 22, 23]:
				X = dict_Xcv[mymin[i]][0]

		clfs[i].fit(X,y[:,i])


	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	
	return clfs


############# PREDICTION #############
def predict(clf, test, y=None):
	print('Prediction...', end=' ')
	timestamp1 = time.time()

	test_prediction = clf.predict(test)

	mylist = []
	for i in range(0, len(test_prediction)):
		for j in range(0, len(test_prediction[1])):
			if test_prediction[i,j]<0:
				mylist.append(max(0, test_prediction[i,j]))
			elif test_prediction[i,j]>1:
				mylist.append(min(1, test_prediction[i,j]))
			else:
				mylist.append(test_prediction[i,j])

	new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	test_prediction = np.array(new_list)

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	if y is not None:
		print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction-y)**2)/ (test.shape[0]*24.0))))

	return test_prediction


def predict2(clfs, test, y=None):
	print('Prediction...', end=' ')
	timestamp1 = time.time()

	test_prediction = np.zeros(test.shape[0]*len(clfs)).reshape(test.shape[0], len(clfs))
	for i, clf in enumerate(clfs):
		test_prediction[:,i] = clfs[i].predict(test)


	mylist = []
	for i in range(0, len(test_prediction)):
		for j in range(0, len(test_prediction[1])):
			if test_prediction[i,j]<0:
				mylist.append(max(0, test_prediction[i,j]))
			elif test_prediction[i,j]>1:
				mylist.append(min(1, test_prediction[i,j]))
			else:
				mylist.append(test_prediction[i,j])


	new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	test_prediction = np.array(new_list)
	test_prediction1 = np.array(new_list)

#	for i in range(0, len(test_prediction)): # s3 and w1
#		for j in [2, 7]:
#			if j==7:
#				test_prediction[i,j] = min(max(0, 1-(np.sum(test_prediction[i,5:7])+ np.sum(test_prediction[i,8:9]))), 1)
#			elif j==2:
#				test_prediction[i,j] = min(max(0, 1-(np.sum(test_prediction[i,0:2]) + np.sum(test_prediction[i,3:5]))), 1)


	#new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	#test_prediction = np.array(new_list)

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	if y is not None:
		print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction-y)**2)/ (test.shape[0]*24.0))))

	return test_prediction





def predict3old(clfs, test_tfidf, test_cv, y=None, clfsLabels=None):
	print('Prediction...', end=' ')
	timestamp1 = time.time()

#	test_prediction = np.zeros(test_cv.shape[0]*len(clfs)).reshape(test_cv.shape[0], len(clfs))
	for i, clf in enumerate(clfs):
		test=test_tfidf
#		if i>0:
#			test = hstack([test_tfidf, test_prediction[:,0:i]])
		if i in [10, 12, 15, 16, 18, 20, 21, 23]:
			test=test_cv
#			if i>0:
#				test = hstack([test_cv, test_prediction[:,0:i]])


		#test = test.toarray()
#		test_prediction[:,i] = clfs[i].predict(test)
		#del test

#		x = ma.array(test_prediction[:,i])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,i] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,i] = x1.filled(1)


	mylist = []
	for i in range(0, len(test_prediction)):
		for j in range(0, len(test_prediction[1])):
			if test_prediction[i,j]<0:
				mylist.append(max(0, test_prediction[i,j]))
			elif test_prediction[i,j]>1:
				mylist.append(min(1, test_prediction[i,j]))
			else:
				mylist.append(test_prediction[i,j])

	new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	test_prediction = np.array(new_list)
	test_prediction1 = np.array(new_list)



#	for i in range(0, len(test_prediction)): # s3 and w1
#		for j in [2, 7]:
#			if j==7:
#				test_prediction[i,j] = min(max(0, 1-(np.sum(test_prediction[i,5:7])+ np.sum(test_prediction[i,8:9]))), 1)
#			elif j==2:
#				test_prediction[i,j] = min(max(0, 1-(np.sum(test_prediction[i,0:2]) + np.sum(test_prediction[i,3:5]))), 1)


	#new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	#test_prediction = np.array(new_list)


	l = []
#	if y is not None: 
#		clf2 = trainFromLabels(np.hstack([test_prediction[:,0:2], test_prediction[:,3:5]]), y[:,2])
#		test_prediction[:,2] = clf2.predict(np.hstack((test_prediction[:,0:2], test_prediction[:,3:5])))
#
#		x = ma.array(test_prediction[:,2])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,2] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,2] = x1.filled(1)
#
#		clf1 = trainFromLabels(np.hstack([test_prediction[:,0:1], test_prediction[:,2:24]]), y[:,1])
#		test_prediction[:,1] = clf1.predict(np.hstack((test_prediction[:,0:1], test_prediction[:,2:24])))
#
#		x = ma.array(test_prediction[:,1])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,1] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,1] = x1.filled(1)
#
#		clf3 = trainFromLabels(np.hstack([test_prediction[:,0:3], test_prediction[:,4:24]]), y[:,3])
#		test_prediction[:,3] = clf3.predict(np.hstack((test_prediction[:,0:3], test_prediction[:,4:24])))
#
#		x = ma.array(test_prediction[:,3])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,3] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,3] = x1.filled(1)
#
#		clf4 = trainFromLabels(np.hstack([test_prediction[:,0:4], test_prediction[:,5:24]]), y[:,4])
#		test_prediction[:,4] = clf4.predict(np.hstack((test_prediction[:,0:4], test_prediction[:,5:24])))
#
#		x = ma.array(test_prediction[:,4])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,4] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,4] = x1.filled(1)
#
#		clf7 = trainFromLabels(np.hstack([test_prediction[:,5:7], test_prediction[:,8:24]]), y[:,7])
#		test_prediction[:,7] = clf7.predict(np.hstack((test_prediction[:,5:7], test_prediction[:,8:24])))
#
#		x = ma.array(test_prediction[:,7])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,7] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,7] = x1.filled(1)
#
#		clf5 = trainFromLabels(test_prediction[:,6:24], y[:,5])
#		test_prediction[:,5] = clf5.predict(test_prediction[:,6:24])
#
#		x = ma.array(test_prediction[:,5])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,5] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,5] = x1.filled(1)
#
#		clf6 = trainFromLabels(np.hstack([test_prediction[:,5:6], test_prediction[:,7:24]]), y[:,6])
#		test_prediction[:,6] = clf6.predict(np.hstack((test_prediction[:,5:6], test_prediction[:,7:24])))
#
#		x = ma.array(test_prediction[:,6])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,6] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,6] = x1.filled(1)
#
#		l = [clf2, clf7, clf5, clf1, clf3, clf6, clf4]
		#
#	elif len(clfsLabels)>0:
#		test_prediction[:,2] = clfsLabels[0].predict(np.hstack((test_prediction[:,0:2], test_prediction[:,3:5])))
#		x = ma.array(test_prediction[:,2])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,2] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,2] = x1.filled(1)
#
#		test_prediction[:,1] = clfsLabels[3].predict(np.hstack((test_prediction[:,0:1], test_prediction[:,2:24])))
#		x = ma.array(test_prediction[:,1])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,1] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,1] = x1.filled(1)
#
#		test_prediction[:,3] = clfsLabels[4].predict(np.hstack((test_prediction[:,0:3], test_prediction[:,4:24])))
#		x = ma.array(test_prediction[:,3])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,3] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,3] = x1.filled(1)
#
#		test_prediction[:,4] = clfsLabels[6].predict(np.hstack((test_prediction[:,0:4], test_prediction[:,5:24])))
#		x = ma.array(test_prediction[:,4])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,4] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,4] = x1.filled(1)
#
#		test_prediction[:,7] = clfsLabels[1].predict(np.hstack((test_prediction[:,5:7], test_prediction[:,8:24])))
#		x = ma.array(test_prediction[:,7])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,7] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,7] = x1.filled(1)
#
#		test_prediction[:,5] = clfsLabels[2].predict(test_prediction[:,6:24])
#		x = ma.array(test_prediction[:,5])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,5] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,5] = x1.filled(1)
#
#		test_prediction[:,6] = clfsLabels[5].predict(np.hstack((test_prediction[:,5:6], test_prediction[:,7:24])))
#		x = ma.array(test_prediction[:,6])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,6] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,6] = x1.filled(1)
#
#	s1=np.sum(test_prediction[:, 0:5], axis=1)
#	s2=np.sum(test_prediction[:, 5:9], axis=1)
#	test_prediction[:, 0:5] = (test_prediction[:, 0:5].T/s1).T
##	#test_prediction[:, 5:9] = (test_prediction[:, 5:9].T/s2).T
#	s = np.sum(test_prediction[:,6:9], axis=1)
#	a = 1 - s
#	test_prediction[s>1, 5] = a[s>1,:]
#	x = ma.array(test_prediction[:,5])
#	x1 = ma.masked_inside(x, -10000., 0.)
#	test_prediction[:,5] = x1.filled(0)
#	x1 = ma.masked_inside(x, 1., 100000.)
#	test_prediction[:,5] = x1.filled(1)
##
#	s = np.sum(np.hstack((test_prediction[:,9:15], test_prediction[:,16:24])), axis=1)
#	a = 3 - s
#	test_prediction[s>3, 15] = a[s>3,:]
#	x = ma.array(test_prediction[:,15])
#	x1 = ma.masked_inside(x, -10000., 0.)
#	test_prediction[:,15] = x1.filled(0)
#	x1 = ma.masked_inside(x, 1., 100000.)
#	test_prediction[:,15] = x1.filled(1)

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	if y is not None:
		print(np.sum(np.array(test_prediction-y)**2, axis=0)/test.shape[0])
		print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction-y)**2)/ (test.shape[0]*24.0))))
		print()
		print(np.sum(np.array(test_prediction1-y)**2, axis=0)/test.shape[0])
		print('\tTrue Train error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction1-y)**2)/ (test.shape[0]*24.0))))

	return (test_prediction, l)


def predict5(clfs, dict_test_tfidf, dict_test_cv, y=None, clfsLabels=None):
	print('Prediction...', end=' ')
	timestamp1 = time.time()

	mymin = dict(zip([3, 9, 14, 16, 18, 19, 22, 23], [0.4, 0.4, 0.375, 0.4, 0.4, 0.3, 0.375, 0.35]))

	k = 2
	if y!=None:
		k=0
	
	test_prediction = np.zeros(dict_test_cv[0.45][k].shape[0]*len(clfs)).reshape(dict_test_cv[0.45][k].shape[0], len(clfs))

	for i, clf in enumerate(clfs):
		test = dict_test_tfidf[0.45][k]
		if i in [3, 9, 14, 16, 18, 19, 22, 23]:
			test = dict_test_tfidf[mymin[i]][k]
		if i in [10, 12, 15, 16, 18, 20, 21, 23]:
			test = dict_test_cv[0.45][k]
			if i in [3, 9, 14, 16, 18, 19, 22, 23]:
				test = dict_test_cv[mymin[i]][k]

		test_prediction[:,i] = clfs[i].predict(test)


	mylist = []
	for i in range(0, len(test_prediction)):
		for j in range(0, len(test_prediction[1])):
			if test_prediction[i,j]<0:
				mylist.append(max(0, test_prediction[i,j]))
			elif test_prediction[i,j]>1:
				mylist.append(min(1, test_prediction[i,j]))
			else:
				mylist.append(test_prediction[i,j])


	new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	test_prediction = np.array(new_list)

	#for i in range(0, len(test_prediction)): # s3 and w1
	#	for j in [2, 7]:
	#		if j==7:
	#			test_prediction[i,j] = (0.25*test_prediction[i,j]+ 0.75*min(max(0, 1-(np.sum(test_prediction[i,5:7])+ np.sum(test_prediction[i,8:9]))), 1))
	#		elif j==2:
	#			test_prediction[i,j] = (0.5*test_prediction[i,j] + 0.5*min(max(0, 1-(np.sum(test_prediction[i,0:2]) + np.sum(test_prediction[i,3:5]))), 1))


	#new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	#test_prediction = np.array(new_list)
	l = []

	if y is not None: 
		clf2 = trainFromLabels(hstack([test_prediction[:,0:2], test_prediction[:,3:5]]), y[:,2])
		clf7 = trainFromLabels(hstack([test_prediction[:,5:7], test_prediction[:,8:9]]), y[:,7])
		l = [clf2, clf7]
		
	else:
		test_prediction[:,2] = clfsLabels[0].predict(hstack([test_prediction[:,0:2], test_prediction[:,3:5]]))
		test_prediction[:,7] = clfsLabels[1].predict(hstack([test_prediction[:,5:7], test_prediction[:,8:9]]))

	

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	if y is not None:
		print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction-y)**2)/ (test.shape[0]*24.0))))

	return (test_prediction, l)


def my_custom_loss_func(y, pred):
	return np.sqrt(np.sum(np.array(pred-y)**2)/ y.shape[0])

def trainFromLabels(labelsX, y_true):
	scorefun = make_scorer(my_custom_loss_func, greater_is_better=False)
	tuned_parametersRidge = {'alpha': [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 10, 20, 30, 40, 50, 100]}
	clf = GridSearchCV(linear_model.Ridge(), param_grid=tuned_parametersRidge, cv=5, scoring=scorefun, n_jobs=6)
	clf.fit(labelsX, y_true)

	return clf


def predictFromLabels(clf, labelsTest):
	y_pred = clf.predict(labelsTest)
	return y_pred


############# SAVE PREDICTION RESULTS #############
def saveResults(ofname, test_prediction):
	print('Save prediction results...', end=' ')
	timestamp1 = time.time()

	aHeader = 'id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15'

	#negative values to 0
	#mylist = []
	#for i in range(0, len(test_prediction)):
	#	for j in range(0, len(test_prediction[1])):
	#		if test_prediction[i,j]<0:
	#			mylist.append(max(0, test_prediction[i,j]))
	#		elif test_prediction[i,j]>1:
	#			mylist.append(min(1, test_prediction[i,j]))
	#		else:
	#			mylist.append(test_prediction[i,j])

	#new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	#new_array = np.array(new_list)

	#prediction = np.array(np.hstack([np.matrix(t2['id']).T, new_array]))  

	prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))

	np.savetxt(ofname, [aHeader], fmt='%s')

	col = '%i,' + '%f,'*23 + '%f'
	f_handle = open(ofname, 'ab')
	np.savetxt(f_handle, prediction, fmt=col, delimiter=',')
	f_handle.close()

	
	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))



############# STATE FEATURE #############
def preprocess_categorical_feature(categorical_column):
	cat = list(set(categorical_column))
	val = range(0,len(cat))
	dictionary = dict(zip(cat, val))

	list_sc = categorical_column.tolist()

	loc_int = [dictionary[list_sc[i]] for i in range(0,len(list_sc))]
	loc_int1 = [loc_int[i:i+1] for i in range(0, len(loc_int), 1)] 

	enc = OneHotEncoder()
	enc.fit(loc_int1)

	new_features = enc.transform(loc_int1)

	return new_features



############# EMOTICONES FEATURE #############
def emoticones_feature(tweets):
	emo_sentiments = []
	for t in tweets:
		emo_sentiments.append(emo.analyze_tweet(t))

	return emo_sentiments


############# STATE FEATURE #############
def preprocess_emoticones_feature(categories, categorical_data):
	val = range(len(categories))
	dictionary = dict(zip(categories, val))

	loc_int = [dictionary[categorical_data[i]] for i in range(0,len(categorical_data))]
	loc_int1 = [loc_int[i:i+1] for i in range(0, len(loc_int), 1)] 

	enc = OneHotEncoder()
	enc.fit(loc_int1)

	new_features = enc.transform(loc_int1)


	return new_features


def sentimentFeatures(tweets, type='both'): #type = 'polarity', 'subjectivity', or 'both'
	print('Sentiment features...', end= ' ')
	timestamp1 = time.time()

	#tb = Blobber(pos_tagger=PerceptronTagger())

	l_tweets = [[t, type] for t in tweets]

	aPool = Pool()
	l =  aPool.map(getSentiment, l_tweets)


	if type=='both':
		lpol = [i[0] for i in l]
		lsub = [i[1] for i in l]

		lpol = [(i+1)/2  for i in lpol]
		l  = [[lpol[i], lsub[i]] for i in range(len(lpol))]

	elif type=='polarity':
		lpol = [i[0] for i in l]
		l = [[(i+1)/2] for i in lpol]


	aPool.close()
	aPool.join()

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	return l


def getSentiment(tweetType): #tweetType = [tweetStr, type]

	tb = Blobber()
	
	return {'both':list(tb(tweetType[0]).sentiment), 
			'polarity':[tb(tweetType[0]).polarity], 
			'subjectivity':[tb(tweetType[0]).subjectivity]}[tweetType[1]]




def new_features2(keywords,tweets):
	""" recieves a vector of strings with the keywords, creates
		a colum for each of the words in the vector. 

		example: keywords = ['tornado','rain','sun']
	"""

	alist = []
	for w in keywords:
		alist.append([int(w in t.lower()) for t in tweets])

	alist = list(map(list, zip(*alist)))

	return np.array(alist)


def new_features3(tweets, aRe=None):

	if aRe == None:
		aRe = re.compile('(\d+\.?\d+)[ \t]*mph', re.IGNORECASE)

	l = len(tweets)*[None]

	for i, t in enumerate(tweets):
		g = re.findall(aRe, t)
		if g:
			if isinstance(g[0], tuple):
				if g[0][2]=='':
					l[i] = float(g[0][0])
				else:
					l[i] = float(g[0][2])
			else:
				l[i] = float(g[0])
		else:
			l[i] = -1

	l_real = [it for it in l if it != -1]
	m = np.mean(l_real)
	l = [[m] if it==-1 else [it] for it in l]
	l = np.array(l)

	l = (l+min(l))/(abs(min(l))+max(l))


	return l


def test_david():
	keywords=["cloud", "cold", "dry", "hot", "humid", "hurricane", 
	"ice", "rain","drizzle", "snow", "storm", "sun", 
	"tornado", "wind","downpour","flood","freeze","hail","lightning",
	"overcast","thunder","typhoon"]

	alist = new_features2(keywords, t['tweet'].tolist())

	return alist




def predictGS(clfs, test, y):
	print('Prediction...', end=' ')
	timestamp1 = time.time()

	test_prediction = np.zeros(test.shape[0]*len(clfs)).reshape(test.shape[0], len(clfs))
	for i, clf in enumerate(clfs):
		test_prediction[:,i] = clfs[i].predict(test)

	mylist = []
	for i in range(0, len(test_prediction)):
		for j in range(0, len(test_prediction[1])):
			if test_prediction[i,j]<0:
				mylist.append(max(0, test_prediction[i,j]))
			elif test_prediction[i,j]>1:
				mylist.append(min(1, test_prediction[i,j]))
			else:
				mylist.append(test_prediction[i,j])

	new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	test_prediction = np.array(new_list)
	rmse = np.sqrt(np.sum(np.array(test_prediction-y)**2)/ (test.shape[0]*24.0))

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	print('\tTrain error: {0:.6f}'.format(rmse))

	return rmse








def gsearch(X, y, nSplits = 3, testSize=0.4):
	
	sss = ShuffleSplit(y.shape[0], n_iter=nSplits, test_size=testSize, random_state=0)

	results = nSplits*[None]
	i = 0
	for train_index, test_index in sss:
		X_train, X_test = X.tocsr()[train_index,:], X.tocsr()[test_index,:]
		y_train, y_test = y[train_index], y[test_index]

		yClusters = clusteringTweets(X_train, y_train, nClusters=100)

		clfs = classifyCluster(X_train, yClusters, y_train, 100)
		y_test_pred = classifyTest(clfs, X_test, 100)
		results[i] = np.sqrt(np.sum(np.array(y_test_pred-y_test)**2)/ (y_test.shape[0]*24.0))
		print('RMSE: {0:.6f}'.format(results[i]))
		i+=1

	print('Average RMSE: {0}'.format(np.mean(results)))

	return results


def gsearchRidge_old(X, Xcv, y, nSplits = 3, testSize=0.4):
	
	sss = ShuffleSplit(y.shape[0], n_iter=nSplits, test_size=testSize, random_state=random.randrange(0, 101))

	results = nSplits*[None]
	i = 0
	for train_index, test_index in sss:
		X_train, X_test = X.tocsr()[train_index,:], X.tocsr()[test_index,:]
		Xcv_train, Xcv_test = Xcv.tocsr()[train_index,:], Xcv.tocsr()[test_index,:]
		y_train, y_test = y[train_index], y[test_index]

		clfs = train3(X_train, Xcv_train, y_train)
		train_prediction, l = predict3(clfs, X_train, Xcv_train, y=y_train)
		test_prediction, l = predict3(clfs, X_test, Xcv_test, clfsLabels=l)


		print(np.sum(np.array(test_prediction-y_test)**2, axis=0)/ y_test.shape[0])
		results[i] = np.sqrt(np.sum(np.array(test_prediction-y_test)**2)/ (y_test.shape[0]*24.0))
		print('\t Out of sample RMSE: {0:.6f}'.format(results[i]))
		print('#########################################################')
		i+=1


	print('Average RMSE: {0}'.format(np.mean(results)))

	return results


def clusteringTweets(X, y, nClusters=5):
	print('Clustering...', end=' ')
	t0 = time.time()

	# pole method
	yc = np.around(y * nClusters, 0)
	yc = yc.astype('int8')

	# Find the possible training matrices for regression
	yregs = yc.shape[1]*[None]
	for j in range(yc.shape[1]):
		yclusters = np.ones(shape=(yc.shape[0], nClusters+1))
		for i in range(nClusters+1):
			yclusters[:,i] = i*(yc[:,j]==i)
		yregs[j] = np.sum(yclusters, axis=1)

	print("{0:.2f} seconds".format(time.time() - t0))

	return yregs



def classifyCluster(X, yClusters, y, nClusters):
	print('Classify clusters...')
	t0 = time.time()

	scorefun = make_scorer(my_custom_loss_func, greater_is_better=False)
	tuned_parameters={'estimator__C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000]}


	#classif = OneVsRestClassifier(svm.SVC(C=0.1, kernel='poly', degree=3, gamma=0.2, class_weight=None, max_iter=-1, random_state=None))
	classif = OneVsRestClassifier(LinearSVC(C=80, random_state=None))
	list_acc = len(yClusters)*[None]
	list_rmse = len(yClusters)*[None]
	clfs = len(yClusters)*[None]
	for i in range(len(yClusters)):
		t1 = time.time()
		#clfs[i] = GridSearchCV(classif, param_grid=tuned_parameters, cv=3, scoring=scorefun, n_jobs=6)
		clfs[i] = OneVsRestClassifier(LinearSVC(C=80, random_state=None))
		clfs[i].fit(X, yClusters[i].astype('int8'))
		y_pred = clfs[i].predict(X)
		list_acc[i] = accuracy_score(yClusters[i], y_pred)
		y_pred2 = y_pred/nClusters;
		list_rmse[i] = np.sum(np.array(y_pred2-y[:,i])**2)/ y.shape[0]

		#print('\tAccuracy label {0}: {1:.2f} ({2:.2f} seconds)'.format(i, list_acc[i], time.time() - t1))
		#print('\tError label {0}: {1:.6f} ({2:.2f} seconds)'.format(i, list_rmse[i], time.time() - t1))

	print('###################################################')
	print('Average accuracy classification: {0:.2f}'.format(np.mean(list_acc)))
	print('RMSE: {0:.6f}'.format(np.sqrt(np.mean(list_rmse))))
	print('###################################################')
	print()
	print('Elapsed time: {0:.2f} seconds'.format(time.time() - t0))

	print(list_rmse)
	return clfs



def classifyTest(clfs, test, nClusters):

	y_pred = np.zeros(shape=(test.shape[0], len(clfs)))
	y_pred_reg = np.zeros(shape=(test.shape[0], len(clfs)))
	for i in range(len(clfs)):
		y_pred[:,i] = clfs[i].predict(test)
		y_pred_reg[:,i] = y_pred[:,i]/nClusters

	return (y_pred, y_pred_reg)
		

def trainClusters(X, y, yClusters, nClusters):
	clfs = yClusters.shape[1]*[None]
	for i in range(yClusters.shape[1]):
		clfs1 = (nClusters+1)*[None]
		for j in range(nClusters+1):
			if np.array(yClusters[:, i]==j).nonzero()[0].shape[0]>0:
				X1 = X.tocsr()[np.array(yClusters[:, i]==j).nonzero()[0],:]
				y1 = y[np.array(yClusters[:, i]==j).nonzero()[0],i]
				clfs1[j] = train_cl(X1, y1)

		clfs[i] = clfs1

	return clfs




def train_cl(X, y):
	#print('Training...', end=' ')
	timestamp1 = time.time()

	my_custom_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)
	
	tuned_parametersRidge = {'alpha': [2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15, 20, 25, 35, 40, 50, 100, 200]}
	clf = None
	if y.shape[0]>2:
		clf = GridSearchCV(linear_model.Ridge(), param_grid=tuned_parametersRidge, cv=3, scoring=my_custom_scorer, n_jobs=6)
	else:
		clf = linear_model.Ridge(alpha = 5)
	clf.fit(X,y)

	timestamp2 = time.time()
	#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	
	return clf



def predictClusters(clfs, yClusters, nClusters, X, y=None):

	l = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
	prediction = np.ones(shape=(X.shape[0], yClusters.shape[1]))
	for i in range(yClusters.shape[1]):
		for j in range(nClusters+1):
			if np.array(yClusters[:, i]==j).nonzero()[0].shape[0]>0:
				X1 = X.tocsr()[np.array(yClusters[:, i]==j).nonzero()[0],:]
				y1_pred = clfs[i][j].predict(X1)

				x = ma.array(y1_pred)
				x1 = ma.masked_inside(x, -1000., 0.)
				y1_pred = x1.filled(0)
				x1 = ma.masked_inside(x, 1., 1000.)
				y1_pred = x1.filled(1)

				prediction[np.array(yClusters[:, i]==j), i] = y1_pred

				#if y is not None:
				#	y1 = y[np.array(yClusters[:, i]==j).nonzero()[0],i]
				#	print('Error label {0} in cluster {1}: {2:.6f}'.format(i, j, np.sum(np.array(y1_pred-y1)**2)/ y1.shape[0]))

			
	return prediction





def trainX(Xtfidf, Xcv, y,method=1):
	myd = {4:'Linear_Reg', 1:'Ridge', 2:'Lasso', 3:'EN'}
	print("# Tuning hyper-parameters for method {0}".format(myd[method]))
	print()
	print('Training...', end=' ')
	timestamp1 = time.time()

	best_alphasTFIDF = [7.5, 1.5, 2.5, 2, 2, 3, 2.5, 4.5, 2, 3, 2.5, 3, 2.5, 1.5, 5, 3, 1.5, 7.5, 2, 4.5, 4.5, 3, 2.5, 1.5]
	best_alphasCV = [100, 20, 50, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 20, 10, 50, 20, 100, 20, 50, 50, 50, 20, 20]
	

	#scorefun = 'mean_squared_error'
	scorefun = make_scorer(my_custom_loss_func, greater_is_better=False)

	# Set the parameters by cross-validation
	tuned_parametersLinearReg = {'fit_intercept': [True, False]}
	tuned_parametersRidge = {'alpha': [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 7.5, 10, 20, 50, 100]}
	tuned_parametersLasso = {'alpha': [0.1, 1, 5, 10, 15, 20, 25, 35, 40, 50, 100]}
	tuned_parametersEN = {'alpha': [0.1, 1, 5, 10, 15, 20, 25, 35, 40, 50, 100], 'l1_ratio': [0.1, 0.5, 0.9]}
	
	
	
	

	test_prediction = np.zeros(shape =y.shape)
	clfs=y.shape[1]*[None]
	for i in range(y.shape[1]):

		#clfs[i] = linear_model.Ridge(alpha = best_alphasTFIDF[i])
		if method==1:
			clfs[i] = GridSearchCV(linear_model.Ridge(), param_grid=tuned_parametersRidge, cv=3, scoring=scorefun, n_jobs=-2)
		elif method==2:
			clfs[i] = GridSearchCV(linear_model.Lasso(), param_grid=tuned_parametersLasso, cv=3, scoring=scorefun, n_jobs=-2)
		elif method==3:
			clfs[i] = GridSearchCV(linear_model.ElasticNet(), param_grid=tuned_parametersEN, cv=3, scoring=scorefun, n_jobs=-2)
		elif method==4:
			clfs[i] = GridSearchCV(linear_model.LinearRegression(), param_grid=tuned_parametersLinearReg, cv=3, scoring=scorefun, n_jobs=-2)

		X = Xtfidf
		if i>0:
			X = hstack([Xtfidf, test_prediction[:,0:i]])
		if i in [10, 12, 15, 16, 18, 20, 21, 23]:
			#clfs[i] = linear_model.Ridge(alpha = best_alphasCV[i])
			X=Xcv
			if i>0:
				X = hstack([Xcv, test_prediction[:,0:i]])

		clfs[i].fit(X,y[:,i])
		test_prediction[:,i] = clfs[i].predict(X)
		x = ma.array(test_prediction[:,i])
		x1 = ma.masked_inside(x, -10000., 0.)
		test_prediction[:,i] = x1.filled(0)
		x1 = ma.masked_inside(x, 1., 100000.)
		test_prediction[:,i] = x1.filled(1)


	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	
	return clfs


def gsearchRidgeX(X, Xcv, y, nSplits = 2, testSize=0.4):
	
	sss = ShuffleSplit(y.shape[0], n_iter=nSplits, test_size=testSize, random_state=0)#random.randrange(0, 101))

	resultsAll = []
	clfsAll = []
	for k in [4, 3]:
		results = nSplits*[None]
		i = 0
		for train_index, test_index in sss:
			X_train, X_test = X.tocsr()[train_index,:], X.tocsr()[test_index,:]
			Xcv_train, Xcv_test = Xcv.tocsr()[train_index,:], Xcv.tocsr()[test_index,:]
			y_train, y_test = y[train_index], y[test_index]

			clfs = trainX(X_train, Xcv_train, y_train, method=k)
			clfsAll.append(clfs)
			train_prediction, l = predict3(clfs, X_train, Xcv_train, y=y_train)
			test_prediction, l = predict3(clfs, X_test, Xcv_test, clfsLabels=l)


			print(np.sum(np.array(test_prediction-y_test)**2, axis=0)/ y_test.shape[0])
			results[i] = np.sqrt(np.sum(np.array(test_prediction-y_test)**2)/ (y_test.shape[0]*24.0))
			print('\t Out of sample RMSE: {0:.6f}'.format(results[i]))
			print('#########################################################')
			i+=1


		print('Average RMSE: {0}'.format(np.mean(results)))
		print()
		print('#########################################################')
		print('#########################################################')
		print('#########################################################')
		print()
		print()

		resultsAll.append(results)

	return (clfsAll, resultsAll)



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
def gsearchRidge(X, Xcv, y, nSplits = 2, testSize=0.4, atfidf=1):
	
	sss = ShuffleSplit(y.shape[0], n_iter=nSplits, test_size=testSize, random_state=random.randrange(0, 101))

	results = nSplits*[None]
	errors = nSplits*[None]
	i = 0
	for train_index, test_index in sss:
		X_train, X_test = X.tocsr()[train_index,:], X.tocsr()[test_index,:]
		Xcv_train, Xcv_test = Xcv.tocsr()[train_index,:], Xcv.tocsr()[test_index,:]
		y_train, y_test = y[train_index], y[test_index]

		clfs = train3(X_train, Xcv_train, y_train, atfidf)
		train_prediction, l = predict3(clfs, X_train, Xcv_train, y=y_train)
		test_prediction, l = predict3(clfs, X_test, Xcv_test, clfsLabels=l)

		errors[i] = np.sum(np.array(test_prediction-y_test)**2, axis=0)/ y_test.shape[0]

		print(np.sum(np.array(test_prediction-y_test)**2, axis=0)/ y_test.shape[0])
		results[i] = np.sqrt(np.sum(np.array(test_prediction-y_test)**2)/ (y_test.shape[0]*24.0))
		print('\t Out of sample RMSE: {0:.6f}'.format(results[i]))
		print('#########################################################')
		i+=1


	s = np.zeros(shape=(1, 24))
	for i in range(nSplits):
		s += errors[i]
	s = s/nSplits
	print(s)

	print('Average RMSE: {0}'.format(np.mean(results)))

	return (results, s)
	


def train3(Xtfidf, Xcv, y, atfidf=1):
	print('Training...', end=' ')
	timestamp1 = time.time()

	best_alphasTFIDF = [7.5, 1.5, 2.5, 2, 2, 3, 2.5, 4.5, 2, 3, 2.5, 3, 2.5, 1.5, 5, 3, 1.5, 7.5, 2, 4.5, 4.5, 3, 2.5, 1.5]
	best_alphasCV = [100, 20, 50, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 20, 10, 50, 20, 100, 20, 50, 50, 50, 20, 20]

	my_custom_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)
	tuned_parametersSGD = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 
						   'penalty' : ['l2', 'l1', 'elasticnet'], 'epsilon':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]}
	
	clfs=y.shape[1]*[None]
	for i in range(y.shape[1]):

		#clfs[i] = linear_model.Ridge(alpha = best_alphasTFIDF[i])
		clfs[i] = GridSearchCV(linear_model.SGDRegressor(), param_grid=tuned_parametersSGD, cv=3, scoring=my_custom_scorer, n_jobs=-2)
		if atfidf !=1:
			#clfs[i] = linear_model.Ridge(alpha = best_alphasCV[i])
			clfs[i] = GridSearchCV(linear_model.SGDRegressor(), param_grid=tuned_parametersSGD, cv=3, scoring=my_custom_scorer, n_jobs=-2)
	
		X = Xtfidf

		clfs[i].fit(X,y[:,i])

		print("Grid Search: best score and params for label {0}:".format(i))
		print(clfs[i].best_params_)
		print()

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	
	return clfs


def predict3(clfs, test_tfidf, test_cv, y=None, clfsLabels=None):
	print('Prediction...', end=' ')
	timestamp1 = time.time()

	test_prediction = np.zeros(test_cv.shape[0]*len(clfs)).reshape(test_cv.shape[0], len(clfs))
	for i, clf in enumerate(clfs):
		test=test_tfidf
#		if i>0:
#			test = hstack([test_tfidf, test_prediction[:,0:i]])
		#if i in [10, 12, 15, 16, 18, 20, 21, 23]:
		#	test=test_cv
#			if i>0:
#				test = hstack([test_cv, test_prediction[:,0:i]])


		#test = test.toarray()
		test_prediction[:,i] = clfs[i].predict(test)
		#del test

#		x = ma.array(test_prediction[:,i])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,i] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,i] = x1.filled(1)


#	mylist = []
#	for i in range(0, len(test_prediction)):
#		for j in range(0, len(test_prediction[1])):
#			if test_prediction[i,j]<0:
#				mylist.append(max(0, test_prediction[i,j]))
#			elif test_prediction[i,j]>1:
#				mylist.append(min(1, test_prediction[i,j]))
#			else:
#				mylist.append(test_prediction[i,j])
#
#	new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
#	test_prediction = np.array(new_list)
#	test_prediction1 = np.array(new_list)



#	for i in range(0, len(test_prediction)): # s3 and w1
#		for j in [2, 7]:
#			if j==7:
#				test_prediction[i,j] = min(max(0, 1-(np.sum(test_prediction[i,5:7])+ np.sum(test_prediction[i,8:9]))), 1)
#			elif j==2:
#				test_prediction[i,j] = min(max(0, 1-(np.sum(test_prediction[i,0:2]) + np.sum(test_prediction[i,3:5]))), 1)


	#new_list = [mylist[i:i+len(test_prediction[1])] for i in range(0, len(mylist), len(test_prediction[1]))]
	#test_prediction = np.array(new_list)


	l = []
#	if y is not None: 
#		clf2 = trainFromLabels(np.hstack([test_prediction[:,0:2], test_prediction[:,3:5]]), y[:,2])
#		test_prediction[:,2] = clf2.predict(np.hstack((test_prediction[:,0:2], test_prediction[:,3:5])))
#
#		x = ma.array(test_prediction[:,2])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,2] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,2] = x1.filled(1)
#
#		clf1 = trainFromLabels(np.hstack([test_prediction[:,0:1], test_prediction[:,2:24]]), y[:,1])
#		test_prediction[:,1] = clf1.predict(np.hstack((test_prediction[:,0:1], test_prediction[:,2:24])))
#
#		x = ma.array(test_prediction[:,1])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,1] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,1] = x1.filled(1)
#
#		clf3 = trainFromLabels(np.hstack([test_prediction[:,0:3], test_prediction[:,4:24]]), y[:,3])
#		test_prediction[:,3] = clf3.predict(np.hstack((test_prediction[:,0:3], test_prediction[:,4:24])))
#
#		x = ma.array(test_prediction[:,3])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,3] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,3] = x1.filled(1)
#
#		clf4 = trainFromLabels(np.hstack([test_prediction[:,0:4], test_prediction[:,5:24]]), y[:,4])
#		test_prediction[:,4] = clf4.predict(np.hstack((test_prediction[:,0:4], test_prediction[:,5:24])))
#
#		x = ma.array(test_prediction[:,4])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,4] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,4] = x1.filled(1)
#
#		clf7 = trainFromLabels(np.hstack([test_prediction[:,5:7], test_prediction[:,8:24]]), y[:,7])
#		test_prediction[:,7] = clf7.predict(np.hstack((test_prediction[:,5:7], test_prediction[:,8:24])))
#
#		x = ma.array(test_prediction[:,7])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,7] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,7] = x1.filled(1)
#
#		clf5 = trainFromLabels(test_prediction[:,6:24], y[:,5])
#		test_prediction[:,5] = clf5.predict(test_prediction[:,6:24])
#
#		x = ma.array(test_prediction[:,5])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,5] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,5] = x1.filled(1)
#
#		clf6 = trainFromLabels(np.hstack([test_prediction[:,5:6], test_prediction[:,7:24]]), y[:,6])
#		test_prediction[:,6] = clf6.predict(np.hstack((test_prediction[:,5:6], test_prediction[:,7:24])))
#
#		x = ma.array(test_prediction[:,6])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,6] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,6] = x1.filled(1)
#
#		l = [clf2, clf7, clf5, clf1, clf3, clf6, clf4]
		#
#	elif len(clfsLabels)>0:
#		test_prediction[:,2] = clfsLabels[0].predict(np.hstack((test_prediction[:,0:2], test_prediction[:,3:5])))
#		x = ma.array(test_prediction[:,2])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,2] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,2] = x1.filled(1)
#
#		test_prediction[:,1] = clfsLabels[3].predict(np.hstack((test_prediction[:,0:1], test_prediction[:,2:24])))
#		x = ma.array(test_prediction[:,1])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,1] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,1] = x1.filled(1)
#
#		test_prediction[:,3] = clfsLabels[4].predict(np.hstack((test_prediction[:,0:3], test_prediction[:,4:24])))
#		x = ma.array(test_prediction[:,3])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,3] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,3] = x1.filled(1)
#
#		test_prediction[:,4] = clfsLabels[6].predict(np.hstack((test_prediction[:,0:4], test_prediction[:,5:24])))
#		x = ma.array(test_prediction[:,4])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,4] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,4] = x1.filled(1)
#
#		test_prediction[:,7] = clfsLabels[1].predict(np.hstack((test_prediction[:,5:7], test_prediction[:,8:24])))
#		x = ma.array(test_prediction[:,7])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,7] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,7] = x1.filled(1)
#
#		test_prediction[:,5] = clfsLabels[2].predict(test_prediction[:,6:24])
#		x = ma.array(test_prediction[:,5])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,5] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,5] = x1.filled(1)
#
#		test_prediction[:,6] = clfsLabels[5].predict(np.hstack((test_prediction[:,5:6], test_prediction[:,7:24])))
#		x = ma.array(test_prediction[:,6])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,6] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,6] = x1.filled(1)
#
#	s1=np.sum(test_prediction[:, 0:5], axis=1)
#	s2=np.sum(test_prediction[:, 5:9], axis=1)
#	test_prediction[:, 0:5] = (test_prediction[:, 0:5].T/s1).T
##	#test_prediction[:, 5:9] = (test_prediction[:, 5:9].T/s2).T
#	s = np.sum(test_prediction[:,6:9], axis=1)
#	a = 1 - s
#	test_prediction[s>1, 5] = a[s>1,:]
#	x = ma.array(test_prediction[:,5])
#	x1 = ma.masked_inside(x, -10000., 0.)
#	test_prediction[:,5] = x1.filled(0)
#	x1 = ma.masked_inside(x, 1., 100000.)
#	test_prediction[:,5] = x1.filled(1)
##
#	s = np.sum(np.hstack((test_prediction[:,9:15], test_prediction[:,16:24])), axis=1)
#	a = 3 - s
#	test_prediction[s>3, 15] = a[s>3,:]
#	x = ma.array(test_prediction[:,15])
#	x1 = ma.masked_inside(x, -10000., 0.)
#	test_prediction[:,15] = x1.filled(0)
#	x1 = ma.masked_inside(x, 1., 100000.)
#	test_prediction[:,15] = x1.filled(1)

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	if y is not None:
		print(np.sum(np.array(test_prediction-y)**2, axis=0)/test.shape[0])
		print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction-y)**2)/ (test.shape[0]*24.0))))
		print()
#		print(np.sum(np.array(test_prediction1-y)**2, axis=0)/test.shape[0])
#		print('\tTrue Train error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction1-y)**2)/ (test.shape[0]*24.0))))

	return (test_prediction, l)



def cropBoundaries(aPred, rangeMin=(-10000.0, 0.), rangeMax=(1., 10000.0)):

	x = ma.array(aPred)
	x1 = ma.masked_inside(x, rangeMin[0], rangeMin[1])
	new_aPred = x1.filled(rangeMin[1])
	x1 = ma.masked_inside(x, rangeMax[0], rangeMax[1])
	new_aPred = x1.filled(rangeMax[0])

	return new_aPred


def normalizeCols(aPreds, aMax=1.):
	s1=np.sum(aPreds, axis=1)
	new_aPreds = (aPreds.T/s1).T

	return new_aPreds


############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############
############# NEW FEATURE EXTRACTION #############

def correctPred(aPred, type):
	errorsType1 = np.array([0.01198053, 0.04848512, 0.04167657, 0.04270948, 0.03675694])
	errorsType2 = np.array([0.07023812, 0.02702401, 0.04882508, 0.02024568])
	errorsType3 = np.array([0.00383079, 0.01633003, 0.0022736, 0.02176596, 0.00388542, 0.00033479, 0.05006448, 0.00127921, 
						0.0192496, 0.00991813, 0.0049496, 0.01035182, 0.01574303, 0.00181053, 0.00511595])


	mul = 1
	if type==1:
		errorSum=np.sum(errorsType1)
		mul = 1-(errorsType1/errorSum)
	elif type==2:
		errorSum=np.sum(errorsType2)
		mul = 1-(errorsType2/errorSum)
	elif type==3:
		errorSum=np.sum(errorsType3)
		mul = 1-(errorsType3/errorSum)

	return mul*aPred



def new_extractFeatures(train_tweets, test_tweets):
	timestamp1 = time.time()

	dictBest = {0:(2,45000,0.45), 1:(2,55000,0.45), 2:(2,40000,0.5),
					3:(2,50000,0.35), 4:(2,55000,0.45), 5:(2,55000,0.55),
					6:(2,20000,0.4), 7:(2,60000,0.4), 8:(2,60000,0.5),
					9:(2,12500,0.45), 10:(1,40000,0.5), 11:(1,16000,0.3),
					12:(1,65000,0.4), 13:(2,60000,0.35), 14:(2,20000,0.55),
					15:(1,65000,0.55), 16:(2,60000,0.3), 17:(2,30000,0.45),
					18:(1,55000,0.5), 19:(2,50000,0.55), 20:(1,16000,0.55),
					21:(1,55000,0.3), 22:(2,17500,0.55), 23:(1,17500,0.3)}

	emo_train, emo_test, sent_train, sent_test, k_train, k_test, w_train, w_test = new_extract_addicional_features()

	dictFeatures_train = {}
	dictFeatures_test = {}
	aY =None
	for i in range(len(dictBest)):
		X, aY, test = new_feature_extraction(train_tweets, test_tweets, method=dictBest[i][0], maxFeatures=dictBest[i][1], maxdf=dictBest[i][2])
		X = hstack([X, emo_train, sent_train, k_train, w_train])
		test = hstack([test, emo_test, sent_test, k_test, w_test])
		dictFeatures_train[i] = X
		dictFeatures_test[i] = test

	timestamp2 = time.time()
	print('Total time: {0:.2f} seconds'.format(timestamp2 - timestamp1))
	return (dictFeatures_train, dictFeatures_test, aY)




def new_feature_extraction(train_tweets, test_tweets, maxFeatures=35000, ngrams=(1,3), maxdf=0.95, mindf=0.0, isbinary=True, method=1):
	print('Feature extraction...', end=' ')
	timestamp1 = time.time()

	tfidf = None
	if method ==1:
		tfidf = CountVectorizer(decode_error=u'ignore', charset_error=None, strip_accents='unicode', lowercase=True, ngram_range=ngrams, analyzer=u'word', max_df=maxdf, min_df=mindf, max_features=maxFeatures, binary=isbinary)
	else:
		tfidf = TfidfVectorizer(decode_error=u'ignore', strip_accents='unicode', analyzer=u'word', lowercase=True, ngram_range=ngrams, max_df=maxdf, min_df=mindf, max_features=maxFeatures, binary=isbinary)

	tfidf.fit(train_tweets)
	X = tfidf.transform(train_tweets)
	y = np.array(t.ix[:,4:])
	test = tfidf.transform(test_tweets)

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	return (X, y, test)



def new_extract_addicional_features():
	l = emoticones_feature(t.ix[:,1].tolist())
	emo_feature_train = preprocess_emoticones_feature(["NA", "HAPPY", "SAD"], l)

	l = emoticones_feature(t2.ix[:,1].tolist())
	emo_feature_test = preprocess_emoticones_feature(["NA", "HAPPY", "SAD"], l)

	sent_feature_train = sentimentFeatures(t.ix[:,1].tolist(), type='both')	
	sent_feature_test = sentimentFeatures(t2.ix[:,1].tolist(), type='both')

	keywords=['drizzle', 'flood','cloud', 'swarm', 'fog', 'gloom', 'haze', 'mist', 'obfuscate', 
	'obscure', 'overshadow', 'shade', 'rack', 'haziness', 'puff', 
	'billow', 'frost', 'nebula', 'nebulosity', 'vapor', 'veil', 
	'overcast', 'pall', 'brume', 'cold', 'arctic', 'chill', 
	'chilly', 'freezing', 'frigid', 'frigidity', 'frosty', 
	'frozen', 'glacial', 'iciness', 'icy', 'shivery', 'wintry', 
	'crisp', 'gelid', 'polar', 'dry', 'arid', 'juiceless', 
	'barren', 'drained', 'droughty', 'bare', 'dusty', 'hot', 
	'warm', 'heated', 'randy', 'ardent', 'fervent', 'glowing', 
	'burning', 'humid', 'clammy', 'steamy', 'hurricane', 
	'typhoon', 'cyclone', 'ice', 'rain', 'shower', 
	'wet', 'downpour', 'snow', 'storm', 'tempest', 'squall', 
	'gale', 'sun', 'tornado', 'whirlwind', 'twister', 'blow', 
	'wind', 'breeze', 'draught', 'draft', 'mistral', 
	'gust', 'blast', 'flurry', 'whisk', 'whiff', 
	'flutter', 'wafting', 'sirocco', 'hail','lightning']

	kind_train = new_features2(keywords,t.ix[:,1].tolist())
	kind_test = new_features2(keywords,t2.ix[:,1].tolist())

	# append wind 
	aReWind = re.compile('(\d+\.?\d+)[ \t]*mph', re.IGNORECASE)
	wind_train = new_features3(t.ix[:,1].tolist(), aReWind)
	wind_test = new_features3(t2.ix[:,1].tolist(), aReWind)

	return (emo_feature_train, emo_feature_test, sent_feature_train, sent_feature_test, kind_train, kind_test, wind_train, wind_test)



def new_train3(dictFeatures, y, aIndex=None):
	print('Training...', end=' ')
	timestamp1 = time.time()

	y_train = y
	if aIndex is not None:
		y_train = y[aIndex]

	dictBest = {0:(2,45000,0.45), 1:(2,55000,0.45), 2:(2,40000,0.5),
					3:(2,50000,0.35), 4:(2,55000,0.45), 5:(2,55000,0.55),
					6:(2,20000,0.4), 7:(2,60000,0.4), 8:(2,60000,0.5),
					9:(2,12500,0.45), 10:(1,40000,0.5), 11:(1,16000,0.3),
					12:(1,65000,0.4), 13:(2,60000,0.35), 14:(2,20000,0.55),
					15:(1,65000,0.55), 16:(2,60000,0.3), 17:(2,30000,0.45),
					18:(1,55000,0.5), 19:(2,50000,0.55), 20:(1,16000,0.55),
					21:(1,55000,0.3), 22:(2,17500,0.55), 23:(1,17500,0.3)}

	best_alphasTFIDF = [7.5, 1.5, 2.5, 2, 2, 3, 2.5, 4.5, 2, 3, 2.5, 3, 2.5, 1.5, 5, 3, 1.5, 7.5, 2, 4.5, 4.5, 3, 2.5, 1.5]
	best_alphasCV = [100, 20, 50, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 20, 10, 50, 20, 100, 20, 50, 50, 50, 20, 20]
	
	clfs=y.shape[1]*[None]
	for i in range(y.shape[1]):

		#clfs[i] = linear_model.Ridge(alpha = best_alphasCV[i])
		clfs[i] = linear_model.SGDRegressor()
		if dictBest[i][0] !=1:
			#clfs[i] = linear_model.Ridge(alpha = best_alphasTFIDF[i])
			clfs[i] = linear_model.SGDRegressor()

		X_train = dictFeatures[i]
		if aIndex is not None:
			X_train = dictFeatures[i].tocsr()[aIndex,:]		

		clfs[i].fit(X_train, y_train[:,i])

	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
	
	return clfs


def new_predict3(clfs, dictFeatures, y=None, clfsLabels=None, aIndex=None):
	print('Prediction...', end=' ')
	timestamp1 = time.time()

	dictBest = {0:(2,45000,0.45), 1:(2,55000,0.45), 2:(2,40000,0.5),
					3:(2,50000,0.35), 4:(2,55000,0.45), 5:(2,55000,0.55),
					6:(2,20000,0.4), 7:(2,60000,0.4), 8:(2,60000,0.5),
					9:(2,12500,0.45), 10:(1,40000,0.5), 11:(1,16000,0.3),
					12:(1,65000,0.4), 13:(2,60000,0.35), 14:(2,20000,0.55),
					15:(1,65000,0.55), 16:(2,60000,0.3), 17:(2,30000,0.45),
					18:(1,55000,0.5), 19:(2,50000,0.55), 20:(1,16000,0.55),
					21:(1,55000,0.3), 22:(2,17500,0.55), 23:(1,17500,0.3)}

	test_prediction = np.zeros(dictFeatures[0].shape[0]*len(clfs)).reshape(dictFeatures[0].shape[0], len(clfs))
	if aIndex is not None:
		test_prediction = np.zeros(dictFeatures[0].tocsr()[aIndex,:].shape[0]*len(clfs)).reshape(dictFeatures[0].tocsr()[aIndex,:].shape[0], len(clfs))
	for i, clf in enumerate(clfs):
		X_test = dictFeatures[i]
		if aIndex is not None:
			X_test = dictFeatures[i].tocsr()[aIndex,:]		
		test_prediction[:,i] = clfs[i].predict(X_test)

		if i>8:
			x = ma.array(test_prediction[:,i])
			x1 = ma.masked_inside(x, -10000., 0.)
			test_prediction[:,i] = x1.filled(0)
			x1 = ma.masked_inside(x, 1., 100000.)
			test_prediction[:,i] = x1.filled(1)


	l = []

#	if y is not None: 
#		y_test = y
#		if aIndex is not None:
#			y_test = y[aIndex]	
#
#		clf2 = trainFromLabels(np.hstack([test_prediction[:,0:2], test_prediction[:,3:5]]), y_test[:,2])
#		test_prediction[:,2] = clf2.predict(np.hstack((test_prediction[:,0:2], test_prediction[:,3:5])))
#
#		x = ma.array(test_prediction[:,2])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,2] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,2] = x1.filled(1)
#
#		clf1 = trainFromLabels(np.hstack([test_prediction[:,0:1], test_prediction[:,2:24]]), y_test[:,1])
#		test_prediction[:,1] = clf1.predict(np.hstack((test_prediction[:,0:1], test_prediction[:,2:24])))
#
#		x = ma.array(test_prediction[:,1])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,1] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,1] = x1.filled(1)
#
#		clf3 = trainFromLabels(np.hstack([test_prediction[:,0:3], test_prediction[:,4:24]]), y_test[:,3])
#		test_prediction[:,3] = clf3.predict(np.hstack((test_prediction[:,0:3], test_prediction[:,4:24])))
#
#		x = ma.array(test_prediction[:,3])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,3] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,3] = x1.filled(1)
#
#		clf4 = trainFromLabels(np.hstack([test_prediction[:,0:4], test_prediction[:,5:24]]), y_test[:,4])
#		test_prediction[:,4] = clf4.predict(np.hstack((test_prediction[:,0:4], test_prediction[:,5:24])))
#
#		x = ma.array(test_prediction[:,4])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,4] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,4] = x1.filled(1)
#
#		clf7 = trainFromLabels(np.hstack([test_prediction[:,5:7], test_prediction[:,8:24]]), y_test[:,7])
#		test_prediction[:,7] = clf7.predict(np.hstack((test_prediction[:,5:7], test_prediction[:,8:24])))
#
#		x = ma.array(test_prediction[:,7])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,7] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,7] = x1.filled(1)
#
#		clf5 = trainFromLabels(test_prediction[:,6:24], y_test[:,5])
#		test_prediction[:,5] = clf5.predict(test_prediction[:,6:24])
#
#		x = ma.array(test_prediction[:,5])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,5] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,5] = x1.filled(1)
#
#		clf6 = trainFromLabels(np.hstack([test_prediction[:,5:6], test_prediction[:,7:24]]), y_test[:,6])
#		test_prediction[:,6] = clf6.predict(np.hstack((test_prediction[:,5:6], test_prediction[:,7:24])))
#
#		x = ma.array(test_prediction[:,6])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,6] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,6] = x1.filled(1)
#
#		l = [clf2, clf7, clf5, clf1, clf3, clf6, clf4]
#		#
#	elif len(clfsLabels)>0:
#		test_prediction[:,2] = clfsLabels[0].predict(np.hstack((test_prediction[:,0:2], test_prediction[:,3:5])))
#		x = ma.array(test_prediction[:,2])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,2] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,2] = x1.filled(1)
#
#		test_prediction[:,1] = clfsLabels[3].predict(np.hstack((test_prediction[:,0:1], test_prediction[:,2:24])))
#		x = ma.array(test_prediction[:,1])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,1] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,1] = x1.filled(1)
#
#		test_prediction[:,3] = clfsLabels[4].predict(np.hstack((test_prediction[:,0:3], test_prediction[:,4:24])))
#		x = ma.array(test_prediction[:,3])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,3] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,3] = x1.filled(1)
#
#		test_prediction[:,4] = clfsLabels[6].predict(np.hstack((test_prediction[:,0:4], test_prediction[:,5:24])))
#		x = ma.array(test_prediction[:,4])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,4] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,4] = x1.filled(1)
#
#		test_prediction[:,7] = clfsLabels[1].predict(np.hstack((test_prediction[:,5:7], test_prediction[:,8:24])))
#		x = ma.array(test_prediction[:,7])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,7] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,7] = x1.filled(1)
#
#		test_prediction[:,5] = clfsLabels[2].predict(test_prediction[:,6:24])
#		x = ma.array(test_prediction[:,5])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,5] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,5] = x1.filled(1)
#
#		test_prediction[:,6] = clfsLabels[5].predict(np.hstack((test_prediction[:,5:6], test_prediction[:,7:24])))
#		x = ma.array(test_prediction[:,6])
#		x1 = ma.masked_inside(x, -10000., 0.)
#		test_prediction[:,6] = x1.filled(0)
#		x1 = ma.masked_inside(x, 1., 100000.)
#		test_prediction[:,6] = x1.filled(1)
#
#	#s1=np.sum(test_prediction[:, 0:5], axis=1)
#	#test_prediction[:, 0:5] = (test_prediction[:, 0:5].T/s1).T
#
#	s = np.sum(np.hstack((test_prediction[:,0:1], test_prediction[:,2:5])), axis=1)
#	a = 1 - s
#	test_prediction[s>1, 1] = a[s>1,:]
#	x = ma.array(test_prediction[:,1])
#	x1 = ma.masked_inside(x, -10000., 0.)
#	test_prediction[:,1] = x1.filled(0)
#	x1 = ma.masked_inside(x, 1., 100000.)
#	test_prediction[:,1] = x1.filled(1)
#
#	s = np.sum(test_prediction[:,6:9], axis=1)
#	a = 1 - s
#	test_prediction[s>1, 5] = a[s>1,:]
#	x = ma.array(test_prediction[:,5])
#	x1 = ma.masked_inside(x, -10000., 0.)
#	test_prediction[:,5] = x1.filled(0)
#	x1 = ma.masked_inside(x, 1., 100000.)
#	test_prediction[:,5] = x1.filled(1)
###
#	s = np.sum(np.hstack((test_prediction[:,9:15], test_prediction[:,16:24])), axis=1)
#	a = 3 - s
#	test_prediction[s>3, 15] = a[s>3,:]
#	x = ma.array(test_prediction[:,15])
#	x1 = ma.masked_inside(x, -10000., 0.)
#	test_prediction[:,15] = x1.filled(0)
#	x1 = ma.masked_inside(x, 1., 100000.)
#	test_prediction[:,15] = x1.filled(1)


	#test_prediction[:, 0:5] = normalize(test_prediction[:, 0:5], norm='l1')
	#test_prediction[:, 5:9] = normalize(test_prediction[:, 5:9], norm='l1')
	#test_prediction[:, 0:5] = scale(test_prediction[:, 0:5], axis=1)
	#test_prediction[:, 5:9] = scale(test_prediction[:, 5:9], axis=1)
	#test_prediction[:, 9:24] = normalize(test_prediction[:, 9:24])

	#test_prediction[:,0:5] = correctPred(test_prediction[:,0:5], 1)
	#test_prediction[:,5:9] = correctPred(test_prediction[:,5:9], 2)
	#test_prediction[:,9:24] = correctPred(test_prediction[:,9:24], 3)


	timestamp2 = time.time()
	print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	if y is not None:
		y_test = y
		if aIndex is not None:
			y_test = y[aIndex]	
		print(np.sum(np.array(test_prediction-y_test)**2, axis=0)/X_test.shape[0])
		print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction-y_test)**2)/ (X_test.shape[0]*24.0))))
		print()
#		print(np.sum(np.array(test_prediction1-y)**2, axis=0)/test.shape[0])
#		print('\tTrue Train error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction1-y)**2)/ (test.shape[0]*24.0))))

	return (test_prediction, l)



def new_gsearchRidge(dictFeatures_train, dictFeatures_test, y, nSplits = 3, testSize=0.4):
	
	sss = ShuffleSplit(y.shape[0], n_iter=nSplits, test_size=testSize, random_state=0)

	results = nSplits*[None]
	errors = nSplits*[None]
	i = 0
	for train_index, test_index in sss:
		y_train, y_test = y[train_index], y[test_index]

		clfs = new_train3(dictFeatures_train, y, aIndex=train_index)
		train_prediction, l = new_predict3(clfs, dictFeatures_train, y=y, aIndex=train_index)


		test_prediction, l = new_predict3(clfs, dictFeatures_train, y=None, aIndex=test_index, clfsLabels=l)

		errors[i] = np.sum(np.array(test_prediction-y_test)**2, axis=0)/ y_test.shape[0]

		print(np.sum(np.array(test_prediction-y_test)**2, axis=0)/ y_test.shape[0])
		results[i] = np.sqrt(np.sum(np.array(test_prediction-y_test)**2)/ (y_test.shape[0]*24.0))
		print('\t Out of sample RMSE: {0:.6f}'.format(results[i]))
		print('#########################################################')
		i+=1


	s = np.zeros(shape=(1, 24))
	for i in range(nSplits):
		s += errors[i]
	s = s/nSplits
	print(s)

	print('Average RMSE: {0}'.format(np.mean(results)))

	return (results, s)




#############################################################################################
#############################################################################################
#############################################################################################
def run_load_data(file):
	t = p.read_csv(file)
	return t

def run_preprocess(data):
	#print('Preprocessing...', end=' ')
	#timestamp1 = time.time()
	

	pr = Preprocess(rm_punctuation=True, rm_special=True, special_words=['mention', 'rt', 'link'], split_compound=True, translate=False, correct=False, singularize=False, 
		lemmatize=False, abbs_file = 'tweeter_abbreviations.csv', expand_abbs=True, substitute_url=True, rm_digits=False, rm_repeated_chars=False, rm_single_chars=False)

	if isinstance(data, str):
		pr_tweets = pr.runParallel([data])
		return pr_tweets[0]

	if isinstance(data, list):
		pr_tweets = pr.runParallel(data)
		return pr_tweets

	data_tweets = data['tweet'].tolist()
	pr_tweets = pr.runParallel(data_tweets)


	#timestamp2 = time.time()
	#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

	return pr_tweets


############# FEATURE EXTRACTION #############
def run_feature_extraction(tweets, pr_tweets, maxFeatures=35000, ngrams=(1,3), maxdf=0.95, mindf=0.0, isbinary=True, method=1, clf=None):
	#print('Feature extraction...', end=' ')
	#timestamp1 = time.time()

	keywords=['drizzle', 'flood','cloud', 'swarm', 'fog', 'gloom', 'haze', 'mist', 'obfuscate', 
	'obscure', 'overshadow', 'shade', 'rack', 'haziness', 'puff', 
	'billow', 'frost', 'nebula', 'nebulosity', 'vapor', 'veil', 
	'overcast', 'pall', 'brume', 'cold', 'arctic', 'chill', 
	'chilly', 'freezing', 'frigid', 'frigidity', 'frosty', 
	'frozen', 'glacial', 'iciness', 'icy', 'shivery', 'wintry', 
	'crisp', 'gelid', 'polar', 'dry', 'arid', 'juiceless', 
	'barren', 'drained', 'droughty', 'bare', 'dusty', 'hot', 
	'warm', 'heated', 'randy', 'ardent', 'fervent', 'glowing', 
	'burning', 'humid', 'clammy', 'steamy', 'hurricane', 
	'typhoon', 'cyclone', 'ice', 'rain', 'shower', 
	'wet', 'downpour', 'snow', 'storm', 'tempest', 'squall', 
	'gale', 'sun', 'tornado', 'whirlwind', 'twister', 'blow', 
	'wind', 'breeze', 'draught', 'draft', 'mistral', 
	'gust', 'blast', 'flurry', 'whisk', 'whiff', 
	'flutter', 'wafting', 'sirocco', 'hail','lightning']

	aReWind = re.compile('(\d+\.?\d+)[ \t]*mph', re.IGNORECASE)


	if clf != None:
		test = clf.transform(pr_tweets)

#		l = emoticones_feature(tweets)
#		emo_feature = preprocess_emoticones_feature(["NA", "HAPPY", "SAD"], l)
#		test = hstack([test, emo_feature])
#
#		sent_feature = sentimentFeatures(tweets, type='both')
#		test = hstack([test, sent_feature])
#		test = hstack([test, new_features2(keywords,tweets)])
#		test = hstack([test, new_features3(tweets, aReWind)])

		return test


	tfidf = None
	if method ==1:
		tfidf = CountVectorizer(decode_error=u'ignore', charset_error=None, strip_accents='unicode', lowercase=True, ngram_range=ngrams, analyzer=u'word', max_df=maxdf, min_df=mindf, max_features=maxFeatures, binary=isbinary)
	else:
		tfidf = TfidfVectorizer(decode_error=u'ignore', strip_accents='unicode', analyzer=u'word', lowercase=True, ngram_range=ngrams, max_df=maxdf, min_df=mindf, max_features=maxFeatures, binary=isbinary)

	tfidf.fit(pr_tweets)
	X = tfidf.transform(pr_tweets)

#	l = emoticones_feature(tweets)
#	emo_feature = preprocess_emoticones_feature(["NA", "HAPPY", "SAD"], l)
#	X = hstack([X, emo_feature])
#
#	sent_feature = sentimentFeatures(tweets, type='both')	
#	X = hstack([X, sent_feature])
#
#	X = hstack([X, new_features2(keywords,tweets)])
#
#	# append wind 
#	X = hstack([X, new_features3(tweets, aReWind)])

	return (tfidf, X)


############# PREDICTION #############
def run_predict(clf, test, y=None):
	test_prediction = clf.predict(test)

	x = ma.array(test_prediction[:,1])
	x1 = ma.masked_inside(x, -10000., 0.)
	test_prediction[:,1] = x1.filled(0)
	x1 = ma.masked_inside(x, 1., 100000.)
	test_prediction[:,1] = x1.filled(1)

	if y is not None:
		print('\tTrain error: {0:.6f}'.format(np.sqrt(np.sum(np.array(test_prediction-y)**2)/ (test.shape[0]*24.0))))

	return test_prediction