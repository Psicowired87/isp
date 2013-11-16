
from Model import *
from Vectorizers import *
import itertools

class SimpleModelFactory:
    ''' Class that generalizes the GANGRENAAAA mehod.
    -------
    Author: Antonio
    Date: 29/10/2013
    ---Last modification:
    Author:
    Date:
    '''

    def __init__(self):
        self.test = {}
    
    def create_models(self): 
        if len(self.test) == 0:
            raise AttributeError("Empty model test list")
            
        models = []
        
        for model in self.test:
            a = [vals for vals in self.test[model].values()]
            try:           
                combinations = list(itertools.product(*a))
            except TypeError:
                msg = "Not iterable object found while computing combinations. Are you shure that if there is a single value in the paramenters it is into parenthesis? " 
                raise AttributeError(msg+str(a))
            
            for c in combinations:
                curr_params = dict(zip(self.test[model].keys(),c))
            
                inst = SimpleModel(model,curr_params)
            
                models.append(inst)
   
        return models
        
class TripleModelFactory:
    ''' Class that generalizes the GANGRENAAAA mehod.
    -------
    Author: Antonio
    Date: 29/10/2013
    ---Last modification:
    Author:
    Date:
    '''

    def __init__(self):
        self.s = {}
        self.w = {}
        self.k = {}
        
    def create_models(self): 
        if len(self.s) == 0 or len(self.w) == 0 or len(self.k) == 0:
            raise AttributeError("Empty model test list")
            
        #for model in self.test:
        inst = TripleModelSearch(s=self.s, w=self.w, k=self.k)
        
        
        return [inst]
        
class SimpleVectorizerFactory(object):
    def __init__(self):
        self.test = {}
    
    def create_vects(self): 
        if len(self.test) == 0:
            raise AttributeError("Empty model test list")
            
        models = []
        for model in self.test:
            a = [vals for vals in self.test[model].values()]
            
            combinations = list(itertools.product(*a))
            
            for c in combinations:
                curr_params = dict(zip(self.test[model].keys(),c))
                
                inst = SimpleVectorizer(model,curr_params)
                
                models.append(inst)
        return models
    