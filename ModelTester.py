from sklearn.cross_validation import KFold
import sklearn.feature_extraction.text
import sklearn.linear_model
import load_data_module
import itertools

""" Framework find the best model and parameters using crossvalidation """

class ModelTester:
	
    
    def search(self,raw_data, vectorizers, models, k):
		   
        perfs = {}
        
        #Data partition integer indices = true because were using scipy sparse
        folds = self.partition(raw_data,k)
        
        for vect in vectorizers:
            for combi in vect.get_combi_params():
                data = combi.fit_transform(raw_data)
                for model in models:
                    perfs.update(model.get_perf(data, combi, folds))
        return perfs
        
    def partition(self, raw_data,k):
        return KFold(raw_data.shape[0], k)
        
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
        for num in self.params['max_features']:
            for mode in self.params['binary']:
                combinations.append(CountVectorizer({"max_features":num,"binary":mode}))
            
        return combinations
                    
                       
class Vectorizer(object):
    """Implements a abstract class for a vectorizer,
    probably unneeded due the duck typing feature of python
    """
    def __init__(self, params):
        self.vct = None
        
    def fit_transform(self,raw_data):
        self.vct.fit_transform(raw_data['tweets'])
    
class CountVectorizer(Vectorizer):
    def __init__(self,params):
        self.vct = sklearn.feature_extraction.text.CountVectorizer(max_features=params["max_features"], binary=params["binary"])
        
    def fit_transform(self, raw_data):
        return self.vct.fit_transform(raw_data)
        
class OualidParanoicCountVectorizer(Vectorizer):
    def __init__(self,params):
        self.vct = sklearn.feature_extraction.text.CountVectorizer(max_features=params["max_features"])
        
    def fit_transform(self, raw_data):
        half1 = raw_data[:len(raw_data)/2]
        half2 = raw_data[len(raw_data)/2:]
        self.vct.fit(half1)
        self.vct.transform(half2)

class ModelFactory:
    def __init__(self, params):
        self.params = params
        
    def get_perf(self, data, vectorizers, folds):
        perfs = {}
        models = self.createModels()
        for model in models:
            perfs.update(model.get_perf(data,vectorizers,folds))
            
        return perfs
            
    def createModels(self):
        pass
        
class RidgeRegressionFactory(ModelFactory):
    def createModels(self):
        models = []
        for alpha in self.params['alpha']:
             models.append(Model(sklearn.linear_model.Ridge(alpha=alpha)))
        return models
       
class Model:
    """
    Respresents the abstract class of all the models
    """
    def __init__(self, clf):
        """
        Gets a dictionary with lists of the parameters to test,f.i:
        RBF -> {gamma:[0.1,0.4,0.9]}
        """
        self.clf = clf
            
    def get_perf(self, data, vect, folds):
        acc = 0
        perfs ={}
        #print("%s (#-features=%d)..." % (clfname, nfeats)) #Warning variables
        i = 0     
        # Cross-validation
        
        for train, test in folds:
            i += 1
            print data
            #Xtrain, Xtest, ytrain, ytest = X.matrix[test], X.matrix[train], y.matrix[test], ymatrix[train] #Warning: depend on X
            Xtrain, Xtest, ytrain, ytest = data.X[train], data.X[test], data.y[train], data.y[test] #Warning: depend on X
        
            self.clf.fit(Xtrain, ytrain)
            ypred = self.clf.predict(Xtest)
            score = self.accuracy_score(ytest, ypred)
            print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score
        acc /= nfolds
        print "## %s (#-features=%d) accuracy=%f" % (clfname, nfeats, acc)
        #self.clf.fit(X,y)
        perfs[(vect,self.clf)] = acc
        return perfs
    
    def accuracy_score(self, ytest,ypred):
        #print ytest.shape, ypred.shape
        size = ytest.shape
        elements = size[0]*size[1]
    #    return sum(sum(np.array(ytest)!=np.array(ypred)))/elements
        #a = np.equal(ytest,ypred)
        return np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
    
class Data(object):
    
    def __init__(self,raw_data):
        self.X = csr_matrix(raw_data['tweets'])
        self.y =  np.array(t.ix[:,4:])
          

raw_data,test = load_data_module.load_data()
tester = ModelTester()

vectorizers = [CountFactory({"max_features":[1000],"binary":[True]})]
models = [RidgeRegressionFactory({"alpha":[1]})]
k = 2
print tester.search(raw_data, vectorizers, models, k)