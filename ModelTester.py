# CARLES CAP DE PC

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
        self.results = dict()

    
    def search(self, vectorizers, models, X, y):
        """ Gets the raw_data and the methods and return the performance using k-fold cv.
        INPUT:
            * raw_data:
            * vectorizers: list of Vectorizer
            * models: list of Model
        """
        #print 'Searching...',
        timestamp1 = time.time()
        perfs = {}

        combinations = self.get_combinations(vectorizers, models)
        #for c in combinations:#print "C",c,"A"
        # You obtain objects instantiated

        # What is e: a tuple of Vectorizer and Model
        for e in combinations:
            features = e[0].fit_transform(X) #TODO Clean this
            #print "Testing", e[0]
            #print "with",e[1]
            clf, acc = e[1].train_test(features,y)  
            perfs[(e[0],clf)] = acc
        #return perfs
        self.results = perfs  #Dictionary with performance
        timestamp2 = time.time()
        #print "%.2f seconds" % (timestamp2 - timestamp1)
        return self.results
       
    def selection(self):
        d= self.results
        return max(d, key=d.get)



    def get_combinations(self, vectorizers, models):
        ''' Return a list of pair of objects Vectorizer and Model
        '''

        a = [vectorizers, models]
        return list(itertools.product(*a))
