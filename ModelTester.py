from sklearn.cross_validation import KFold
import itertools

""" Framework find the best model and parameters using crossvalidation """

class ModelTester:
	""" Main class that takes care of get the raw_data, get the models and return a list with
        the performance of each combination feature selection - model.
    """
    
	def search(raw_data, vectorizers, models, k):
		""" Gets the raw_data and the methods and return the performance using k-fold cv
        """        
        perfs = {}
        
        #Data partition integer indices = true because were using scipy sparse
        folds = partition(raw_data,k)
        
        for vect in vectorizers:
            for combi in vectorizers.get_combi_params()
                data = combi.fit_transform(raw_data)
                for model in models:
                    perf.update(model.get_perf(data, folds))
        
		return perfs
        
class VectorizerFactory(object):
    """
    Implements a Factory for Vectorizers
    """
    def __init__(self,params = None):
        """
        Gets a dictionary with lists of the parameters to test,f.i:
        TF IDF -> {num_of_words:[100,500,1000], minimum_tf:[0.2,0.5,0.6]}
        """
        self.params = params
        
    def get_combi_params(self):
        pass
        


#NUEVO AQUI

class CountFactory(VectorizerFactory):
    """
    Implements a Factory of Count Vectorizers
    """
    
    def get_combi_params(self):
        combinations = []
        for num in self.params['max_words']:
            combinations.append(CountVectorizer({"max_words":num}))
            
        return combinations
                    
                       
class Vectorizer(object):
    """Implements a abstract class for a vectorizer,
    probably unneeded due the duck typing feature of python
    """
    def __init__(self, params):
        self.vct = None
        
    def fit_transform(self,raw_data):
        self.vct.fit_transform(raw_data)
    
class CountVectorizer(Vectorizer):
    def __init__(self,params):
        self.vct = sklearn.CountVectorizer(max_words=params["max_words"])
        
    def fit_transform(self, raw_data):
        return self.vct.fit_transform(raw_data)
        
class OualidParanoicCountVectorizer(Vectorizer):
    def __init__(self,params):
        self.vct = sklearn.CountVectorizer(max_words=params["max_words"])
        
    def fit_transform(self, raw_data):
        half1 = raw_data[:len(raw_data)/2]
        half2 = raw_data[len(raw_data)/2:]
        self.vct.fit(half1)
        self.vct.transform(half2)
        
class Model:
    """
    Respresents the abstract class of all the models
    """
    def __init__(self, clf_class, params = None):
        """
        Gets a dictionary with lists of the parameters to test,f.i:
        RBF -> {gamma:[0.1,0.4,0.9]}
        """
        self.params = params
        self.clf = clf_class
            
    def get_perf(self, data, folds):
        
        acc = 0
        #print("%s (#-features=%d)..." % (clfname, nfeats)) #Warning variables

        # Cross-validation
        for train, test in folds:
            i += 1
            #Xtrain, Xtest, ytrain, ytest = X.matrix[test], X.matrix[train], y.matrix[test], ymatrix[train] #Warning: depend on X
            Xtrain, Xtest, ytrain, ytest = data.X.matrix[train], data.X.matrix[test], data.y.matrix[train], data.y.matrix[test] #Warning: depend on X
            
            self.clf.fit(Xtrain, ytrain)
            ypred = clf.predict(Xtest)
            score = accuracy_score(ytest, ypred)
            print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score
        acc /= nfolds
        print "## %s (#-features=%d) accuracy=%f" % (clfname, nfeats, acc)
        self.clf.fit(X.matrix,y.matrix)
        return acc
        