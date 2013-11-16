

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

### Create Model Factories 
# Those models create the actual models
tri = TripleModelFactory()
tri.s = {
    Ridge:{'alpha':[0.3]},
    #KNeighborsClassifier:{"n_neighbors":[3,1]}
    } #List of Models + params
tri.w = {
    Ridge:{'alpha':[0.3,0.6]},
    #KNeighborsClassifier:{"n_neighbors":[3,1]}
    }
tri.k = {
    Ridge:{'alpha':[0.3]},
    #KNeighborsClassifier:{"n_neighbors":[3,1]}
    }

simple = SimpleModelFactory()
simple.test = {
    Ridge:{'alpha':[0.3]},
    #KNeighborsClassifier:{"n_neighbors":[3,1]}
    } #List of Models + params

modelparams = [tri, simple]

count = SimpleVectorizerFactory()
count.test = {CountVectorizer:{}}

vectparams = [count]
                

#vectparams = {CutoffEntropyVectorizer:{ "cutoff":[1000],"max_n":[2],"labels":[y]}}

searchparams = {"folds":4}

params = Parameters(modelparams,vectparams,searchparams)


############# PARAMETER SELECTION #############
# Class model from Carles implementation.
# Note that model includes CV
models=ModelTester() #include k?
results = models.search(X,y,params)
(vct,clf) = models.selection()

print vct,clf

#search(list_methods, list_parameters,...)
#   initialize model and invokes model.getperformance(...) several times in the search.
#clf =
# return model with best_parameters and best_method


############# PREDICTION #############

# call model.predict(test)
# model is the last one chosen before, thus model has predict function

test = vct.transform(t2['tweet'])

test_prediction = clf.predict(test)
print test_prediction

# return labels, confidence ??? Integrate in clf





############# SAVE RESULTS #############

saveResults(out_name+'.csv', test_prediction,t2)





