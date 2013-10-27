from sklearn.cross_validation import KFold

""" Framework find the best model and parameters using crossvalidation """

class ModelTester:
	""" Main class that takes care of get the data, get the models and return a list with
        the performance of each combination feature selection - model.
    """
    
	def search(data, featureMethods, models, k):
		""" Gets the data and the methods and return the performance
        """        
        perfs = {} #Dictionary to store the results of the folds: Combination -> [res_fold1, res_fold2,...]
        #Data partition
        folds = KFold(X.shape[0], nfolds)
        
        #For each folds
        for f in folds:
            #Test all the combinations
            results = test(f, featureMethods,models)
            #Add each result to the dictionary. If does not exist create it.
            for k in results.keys():
                if k in perfs:
                    perfs[k].expand(results[k])
                else:
                    perfs[k] = [results[k]]
                    
        
        #Create the folds of the data
		return perfs
        
    
    
    def test(fold, feature, models):
        """
        Tests all the combinations of feature selection methods and models using the
        data fold provided
        """
        
        
        pass
        
    

class FeatureSelectionMethod:
    """
    Implements the abstract class of all the Feature Selection Methods
    """
    def __init__(self,params = None):
        """
        Gets a dictionary with lists of the parameters to test,f.i:
        TF IDF -> {num_of_words:[100,500,1000], minimum_tf:[0.2,0.5,0.6]}
        """
        self.params = params
    
    def getdata(self,data):
        """returns the features extracted from the data"""
        return features
    
        
class Model:
    """
    Respresents the abstract class of all the models
    """
    def __init__(self, params = None):
        """
        Gets a dictionary with lists of the parameters to test,f.i:
        RBF -> {gamma:[0.1,0.4,0.9]}
        """
        self.params = params
        
    def getperformance(data):
        """
        Trains the model with the train data, test the model and return the
        performance achieved
        """
        return perf