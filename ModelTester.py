from sklearn.cross_validation import KFold
import itertools
from Factory import *
import time

#from Model import *
#from Data import *

class ModelTester:
    """ Main class that takes care of get the raw_data, get the models and return a list with
        the performance of each combination feature selection - model.
    """

    def __init__(self):
        """ Initialization of the ModelTester.
        INPUT:
            * list_vectorizers: dict
            * vectorizers: list of Vectorizer
            * models: list of Model
            * k
        """
        self.params = None #TODO put warnings for each lazy-init attr for if is None wherever is used
        self.results = dict()

    
    def search(self,X,y, params):
        """ Gets the raw_data and the methods and return the performance using k-fold cv.
        INPUT:
            * raw_data:
            * vectorizers: list of Vectorizer
            * models: list of Model
        """
        print 'Searching...',
        self.params = params
        timestamp1 = time.time()
        perfs = {}

        combinations = self.get_all_combinations()
        #for c in combinations:print "C",c,"A"
        # You obtain objects instantiated

        # What is e: a tuple of Vectorizer and Model
        for e in combinations:
            features = e[0].fit_transform(X) #TODO Clean this
            print "Testing", e[1]
            clf, acc = e[1].train_test(features,y)  
            perfs[(e[0],clf)] = acc
        #return perfs
        self.results = perfs  #Dictionary with performance
        timestamp2 = time.time()
        print "%.2f seconds" % (timestamp2 - timestamp1)
        return self.results
       
    def selection(self):
        d= self.results
        return max(d, key=d.get)

    def get_allmodels(self):
        ''' Return all models in a list'''
        models = []
        for factory in self.params.modelparams:
            models.extend(factory.create_models())
        print models
        return models
        
    
    def get_allvectorizers(self):
        ''' Return all models'''
        vects = []
        for factory in self.params.vectparams:
            vects.extend(factory.create_vects())
            
        return vects
        



    def get_all_combinations(self):
        ''' Return a list of pair of objects Vectorizer and Model
        '''
    

        # Only this in theory:
        models = self.get_allmodels()
        vectorizers = self.get_allvectorizers()
    
        a = [vectorizers,models]
        return list(itertools.product(*a))
