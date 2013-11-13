

# Interns
from load_data_module import *
from sklearn.feature_extraction.text import CountVectorizer
from Parameters import Parameters
from Labels import Labels
from ModelTester import ModelTester
from Vectorizers import *
from Model import *
##from Vectorizer import *
out_name="Count_Ridge"



### LOAD THE DATA ###
# ask for name of the output file NO WAY

# load train as t and test as t2
t,t2 = load_data('../train.csv','../test.csv')

y = Labels(np.array(t.ix[:,4:]))
X = np.array(t['tweet'])


#print t
############# PREPROCESSING STEP - OUALID PUT YOUR THINGS THERE #############



### SET PARAMETERS OF THE PROGRAM ###

#modelparams = {SimpleModel:{'clf_class':[Ridge],{clf_params:{'alpha':[0.5]}}}
modelparams = {TripleModel:{'s_clf_class':[Ridge],'w_clf_class':[Ridge],'k_clf_class':[Ridge],
                's_params':[{'alpha':0.3}],
                'k_params':[{'alpha':0.5}],
                'w_params':[{'alpha':0.7}]}}
                
searchparams = {'k':[2],'nfolds':[2]} ##ANTIONIO, WTF IS K??? AND NFOLDS?? PREVIOUSLY WAS folds, BUT IN MODEL WE WERE USING nfolds LOL
#vectparams = {CutoffEntropyVectorizer:{ "cutoff":[1000],"max_n":[2],"labels":[y]}}
vectparams = {
    SklearnVectorizer:{
        'vct_class':[CountVectorizer],'vct_params':[{'max_df':0.5}]}}
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





