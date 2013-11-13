from sklearn.cross_validation import KFold
import itertools
from Factory import Factory
import time

#from Model import *
from Data import *

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
        factory = Factory()
        combinations = []
    
        list_models = []
        for model in self.params.modelparams:
            a = [vals for vals in self.params.modelparams[model].values()]
            print a
            combinations = list(itertools.product(*a))
            print combinations
            for c in combinations:
                curr_params = dict(zip(self.params.modelparams[model].keys(),c))
                if 'nfolds' not in curr_params and 'nfolds' in self.params.searchparams:
                    curr_params['nfolds'] = self.params.searchparams['nfolds'][0]
                inst = factory.create(model,curr_params)
                
            list_models.append(inst)
        return list_models
    
    def get_allvectorizers(self):
        ''' Return all models'''
        factory = Factory()
        combinations = []
    
        list_vects = []
        for vect in self.params.vectparams:
        
            a = [vals for vals in self.params.vectparams[vect].values()]
        
            combinations = list(itertools.product(*a))
            #combinations = dict(zip(self.vectparams[vect].keys(),combinations))
            list_vects.extend([factory.create(vect,dict(zip(self.params.vectparams[vect].keys(),c))) for c in combinations])
        return list_vects


    def get_all_combinations(self):
        ''' Return a list of pair of objects Vectorizer and Model
        '''
    

        # Only this in theory:
        models = self.get_allmodels()
        vectorizers = self.get_allvectorizers()
    
        a = [vectorizers,models]
        return list(itertools.product(*a))
