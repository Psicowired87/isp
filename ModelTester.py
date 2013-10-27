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
        folds = partitionkfolds(data,k)
        
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
        
    

class FeatureSelectionMethod:
    def __init__(self,params = None):
        self.params = params
    
    def getdata(self):
        return data
        
class Model:
    def __init__(self, params = None):
        self.params = params
        
    def getperformance(data):
        return perf