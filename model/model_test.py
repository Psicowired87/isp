import model
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import numpy as np
from sklearn.metrics import make_scorer

#model.load_data('data/train.csv', 'data/test.csv')
#train_tweets, test_tweets = model.preprocess_data()
#X,y,test, feature_names = model.feature_extraction(train_tweets, test_tweets)
#clf = model.train(X,y)


#test_prediction = model.predict(clf, test)
#model.saveResults('output/absPred.csv', test_prediction)

#if __name__ == "__main__":
#	runTests()
	#model.load_data('data/train.csv', 'data/test.csv')
	#X,y,test = model.feature_selection()
	#clf = model.train(X,y)

	#test_prediction = model.predict(clf, test)
	#model.saveResults('nightX.csv', test_prediction)
def my_custom_loss_func(y, pred):
	return np.sqrt(np.sum(np.array(pred-y)**2)/ (y.shape[0]*24.0))


def gsearch(X,y):
	# Split the dataset: 40% for test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

	# Set the parameters by cross-validation
	tuned_parametersLinearReg = {'fit_intercept': [True, False]}
	tuned_parametersRidge = {'alpha': [0.0001, 0.01, 0.1, 1, 5, 10, 15, 20, 25, 35, 40, 50, 100, 200, 500]}
	tuned_parametersLasso = {'alpha': [0.01, 0.1, 1, 5, 10, 15, 20, 25, 35, 40, 50, 100, 200]}
	tuned_parametersEN = {'alpha': [0.01, 0.1, 1, 5, 10, 15, 20, 25, 35, 40, 50, 100, 200], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
	tuned_parametersSVR = [{'kernel': ['rbf'], 'gamma': [2, 1, 0.5, 0.1, 0.01, 1e-3, 1e-4], 'C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 100]},
	                    {'kernel': ['linear'], 'C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 100]},
	                    {'kernel': ['linear'], 'degree':[2, 3, 4, 5], 'C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 100]}]

	#scorefun = 'mean_squared_error'
	scorefun = make_scorer(my_custom_loss_func, greater_is_better=False)

	#for method, name in enumerate(['Linear_Reg', 'Ridge', 'Lasso', 'EN']):#, 'SVR']):
	for method, name in enumerate(['Ridge']):#, 'SVR']):
		print("# Tuning hyper-parameters for %s" % name)
		print()

		clfs=y_train.shape[1]*[None]
		for i in range(y_train.shape[1]):
			if method==0:
				clfs[i] = GridSearchCV(Ridge(), param_grid=tuned_parametersRidge, cv=5, scoring=scorefun, n_jobs=6)
			elif method==1:
				clfs[i] = GridSearchCV(Lasso(), param_grid=tuned_parametersLasso, cv=5, scoring=scorefun, n_jobs=6)
			elif method==2:
				clfs[i] = GridSearchCV(ElasticNet(), param_grid=tuned_parametersEN, cv=5, scoring=scorefun, n_jobs=6)
			elif method==4:
				clfs[i] = GridSearchCV(SVR(C=1), param_grid=tuned_parametersSVR, cv=5, scoring=scorefun, n_jobs=6)
			elif method==7:
				clfs[i] = GridSearchCV(LinearRegression(), param_grid=tuned_parametersLinearReg, cv=5, scoring=scorefun, n_jobs=6)

			clfs[i].fit(X_train, y_train[:,i])

			print("Grid Search: best score and params for label {0}:".format(i))
			print(clfs[i].best_score_)
			print(clfs[i].best_params_)
			print()

		print("##############################################")
		y_pred = np.zeros(X_test.shape[0]*y_test.shape[1]).reshape(X_test.shape[0], y_test.shape[1])
		for i in range(y_test.shape[1]):
			y_pred[:,i] = clfs[i].predict(X_test)

		print('\tTrain error of {0}: {1:.6f}'.format(name, np.sqrt(np.sum(np.array(y_pred-y_test)**2)/ (X_test.shape[0]*24.0))))
		print("##############################################")
		print()
		print()



def runTests():
	model.load_data('data/train.csv', 'data/test.csv')

	methodPreprocess_list = [1, 2]

	methodFeatureExtraction=[1, 2]
	maxFeatures_list = [16000, 20000, 25000]
	ngrams_list = [(1,3), (1,4), (1,5), (1,2)]
	maxdf_list = [1.0, 0.95, 0.9, 0.85]
	mindf_list = [0.0001]
	binary_list = [True]


	for methodPreprocess in methodPreprocess_list:
		train_tweets, test_tweets = model.preprocess_data(method=methodPreprocess)	
		for maxfeat in maxFeatures_list:
			for ng in ngrams_list:
				for maxd in maxdf_list:
					for mind in mindf_list:
						for bin in binary_list:
							for featureExt in methodFeatureExtraction:
								X,y,test,feature_names = model.feature_extraction(train_tweets, test_tweets, maxFeatures=maxfeat, ngrams=ng, maxdf=maxd, mindf=mind, isbinary=bin, method=featureExt)
								print('\n\n\n')
								print("#########################################################################")
								print("##############################################")
								print('Params preprocessing and features extraction:')
								print('{0}, {1}, {2}, {3}, {4}, {5}, {6}'.format(methodPreprocess, maxfeat, ng, maxd, mind, bin, featureExt))
								print("##############################################")
								gsearch(X,y)
								print("#########################################################################")
								print('\n\n\n')
							



if __name__ == "__main__":
	runTests()




