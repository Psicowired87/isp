from __future__ import print_function
import model
from scipy.sparse import coo_matrix, hstack
#
#model.load_data('data/train.csv', 'data/test.csv')
#train_tweets, test_tweets = model.preprocess_data(method=1)
#dictFeatures_train, dictFeatures_test, y = model.new_extractFeatures(train_tweets, test_tweets)
#results, s = model.new_gsearchRidge(dictFeatures_train, dictFeatures_test, y, nSplits = 3, testSize=0.4)
#
#
#
#clfs = model.new_train3(dictFeatures_train, y)
#train_prediction, l = model.new_predict3(clfs, dictFeatures_train, y=y)
#test_prediction, l = model.new_predict3(clfs, dictFeatures_test, y=None, clfsLabels=l)
#model.saveResults('output/xxx1.csv', test_prediction)


model.load_data('data/train.csv', 'data/test.csv')
train_tweets, test_tweets = model.preprocess_data(method=1)
emo_train, emo_test, sent_train, sent_test, k_train, k_test, w_train, w_test = model.new_extract_addicional_features()

		
l_tfidf = []
l_cv = []

print()
print()
for mF in [12500, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]:
	print('Max Features = {0}'.format(mF))
	for maxd in [0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]:
		print('Max DF = {0}______________________________________________________'.format(maxd))
		X,y,test = model.new_feature_extraction(train_tweets, test_tweets, method=2, maxFeatures=mF, maxdf=maxd, mindf=0.0)
		X = hstack([X, emo_train, sent_train, k_train, w_train])
		test = hstack([test, emo_test, sent_test, k_test, w_test])

		Xcv,ycv,testcv = model.new_feature_extraction(train_tweets, test_tweets, method=1, maxFeatures=mF, maxdf=maxd, mindf=0.0)
		Xcv = hstack([Xcv, emo_train, sent_train, k_train, w_train])
		testcv = hstack([testcv, emo_test, sent_test, k_test, w_test])
		print('----------------------------------------------------------------------')
		print('----------------------------------------------------------------------')
		print('----------------------------------------------------------------------')
		print('TF-IDF')
		res, s = model.gsearchRidge(X, X, y, nSplits=1, testSize=0.45,  atfidf=1)
		l_tfidf.append((res, s))
		print('----------------------------------------------------------------------')
		print('Count Vectorizer')
		res, s = model.gsearchRidge(Xcv, Xcv, ycv, nSplits=1, testSize=0.45,  atfidf=0)
		l_cv.append((res, s))
		print()
		print()
	print('********************************************************************************')
	print('********************************************************************************')
	print()
	print()
	print()
	print()
	


