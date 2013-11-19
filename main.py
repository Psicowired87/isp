

# Interns
from load_data_module import *
from sklearn.feature_extraction.text import CountVectorizer
from Parameters import Parameters
from Labels import Labels
from Factory import *
from ModelTester import ModelTester
from Vectorizers import *
from Model import *
from sklearn.neighbors import KNeighborsClassifier
##from Vectorizer import *
out_name="Count_Ridge"



### LOAD THE DATA ###
# ask for name of the output file NO WAY

# load train as t and test as t2
t,t2 = load_data('./train.csv','./test.csv')

y = Labels(np.array(t.ix[:,4:]))
X = np.array(t['tweet'])


#print t
############# PREPROCESSING STEP - OUALID PUT YOUR THINGS THERE #############



### SET PARAMETERS OF THE PROGRAM ###
factory = Factory()
models = factory.create_models(Ridge, {'alpha': [0.3, 0.5, 0.7], 'fit_intercept': [False, True]})
vects = factory.create_vects(CountVectorizer, {'ngram_range': [(1, 2), (1, 3)], 'max_features': [None, 1000, 100000]})
tester = ModelTester()
results = tester.search(vects, models, X, y)

print results
(vct,clf) = models.selection()
############# PREDICTION #############

# call model.predict(test)
# model is the last one chosen before, thus model has predict function

test = vct.transform(t2['tweet'])

test_prediction = clf.predict(test)
print test_prediction

# return labels, confidence ??? Integrate in clf





############# SAVE RESULTS #############

saveResults(out_name+'.csv', test_prediction,t2)





