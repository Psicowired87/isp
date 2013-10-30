from sklearn.cross_validation import KFold

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
                data = vect.fit_transform(raw_data)
                for model in models:
                    perf.update(model.get_perf(data, folds))
        
		return perfs
        
class Vertorizer:
    """
    Implements the abstract class of all the Feature Selection Methods
    """
    def __init__(self,vct,params = None, fit_tranform_function):
        """
        Gets a dictionary with lists of the parameters to test,f.i:
        TF IDF -> {num_of_words:[100,500,1000], minimum_tf:[0.2,0.5,0.6]}
        """
        self.vct = vct
        self.params = params
        self.fit_transform = fit_tranform_function
    
    def fit_transform(self,raw_data):
        return self.vct.fit_transform(raw_data)
        
        


    
        
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
        