from sklearn.cross_validation import KFold

""" Framework find the best model and parameters using crossvalidation """

class ModelTester:
	""" Main class that takes care of get the data, get the models and return a list with
        the performance of each combination feature selection - model.
    """
    
	def search(data, featureMethods, models, k):
		""" Gets the data and the methods and return the performance using k-fold cv
        """        
        perfs = {} #Dictionary to store the results of the folds: Combination -> [res_fold1, res_fold2,...]
        
        
        
        #Extract features and get a dictionary for each method used -> features
        features = extract(data,featureMethods)
        
        #Data partition integer indices = true because were using scipy sparse
        folds = KFold(X.shape[0], nfolds, indices=True)
        
        #Test all the combinations given the features extracted and the models
        results = evaluate(features, models, folds)
               
        #Add each result to the dictionary. If does not exist create it.
        for combi in results.keys():
            if combi in perfs:
                perfs[combi].expand(results[combi])
            else:
                perfs[combi] = [results[combi]]
                          
        #Create the folds of the data
		return perfs
        
    
    
    def test(self, features, models, folds):
        """
        Tests all the combinations of feature selection methods and models using the
        data fold provided
        """
        results = {}
          
        #For each model test all the feature methods
        for model in models:
            modresults = model.evaluate(features,folds)
            #Add them to the list of results
            results.update(modresults)
        
    def extract(self, data, featureMethods):
        """
        Extract the features using all the feature selection methods given as a parameter with the data
        provided
        """
        features = {}
        for method in featureMethods:
            #For each method ask to get the features from the data
            features[method.name] = method.extract(data)
            
        return features

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
    
    def extract(self,data):
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
        
    def getperformance(self,features,folds):
        """
        Trains the model with the train data, test the model and return the
        performance achieved
        """
        return perf