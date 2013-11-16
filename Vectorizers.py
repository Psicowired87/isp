from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class SimpleVectorizer(object):
    def __init__(self, vct_class, params):
        self.max_n= 1
        self.vct_class = vct_class
        self.vct_params = params
        self.vct = self.vct_class()
        for param in self.vct_params:
            setattr(self.vct, param,self.vct_params[param])
            
        
    def fit(self,raw_data):
        
        self.vct_fitted = self.vct.fit(raw_data)
        
    def transform(self,raw_data):
        return self.vct_fitted.transform(raw_data)
        
    def fit_transform(self, raw_data):
        self.fit(raw_data)
        return self.transform(raw_data)
    
    def get_feature_names(self):
        return self.vct.get_feature_names()
    

class ProbDeviationVectorizer(object):
    
    def __init__(self):
        pass
        
class CutoffEntropyVectorizer(object):
    weights = None
    def __init__(self,vct_class,params):
        self.cutoff = None
        self.labels = None
        self.max_n= 1
        self.params = params
        self.vct = None
        if vct_class is None:
            self.vct_class = CountVectorizer
        try:
            self.vct = self.vct_class()
            for param in self.params:
                setattr(self.vct, param,self.params[param])
            
        except AttributeError:
            raise AttributeError("Error no vct_class found, is it defined at the parameter array?")
        
        
    def fit(self,raw_data):
        self.vct = self.vct.fit(raw_data)
        data = self.vct.transform(raw_data)
        if CutoffEntropyVectorizer.weights is None:
            CutoffEntropyVectorizer.weigths = self.discriminative_measure(data,self.labels.matrix)
        indices = np.argsort(CutoffEntropyVectorizer.weigths)
        self.chosen = indices[:self.cutoff]
        
        
    def transform(self,raw_data):
        data = self.vct.transform(raw_data.X)
        return data[:,self.chosen]
    
    def fit_transform(self, raw_data):
        self.fit(raw_data)
        return self.transform(raw_data)
    
    def get_feature_names(self):
        names = np.array(self.vct.get_feature_names())
        print type(names),type(self.chosen)
        return names[self.chosen]
    
    def discriminative_measure(self,X,y):

        #print X
        #It computes the average value of every feature for the vector of tags (distribution)
        index_matrix = [X[i].indices for i in range(X.shape[1]) ]
        distribution = [sum(y[index_matrix[i]])/len(index_matrix[i]) for i in range(len(index_matrix))]
        # the priors is the average of tags of all the features. It is the definition of a non-discriminative feature
        priors = sum(y)/len(y)
        # We compute the average scaled distance to the priors:
        distance_to_priors = np.array([np.array(distribution[i]-priors) for i in range(len(distribution)) ])
        discriminative = (distance_to_priors>0)*(distance_to_priors/(1-priors)) - (distance_to_priors<0)*(distance_to_priors/(priors))
        return np.sum(discriminative,axis=1)/np.array(y).shape[1]

    
