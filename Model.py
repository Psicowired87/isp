
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier




from sklearn.cross_validation import KFold
import numpy as np

from Data import *


class SimpleModel:
    '''
    Respresents the abstract class of all the individual models.
    -------
    Author: Carles
    Date: 27/10/2013
    ---Last modification:
    Author: Antonio
    Date: 29/10/2013
    '''
    #TODO:
    # Integrate cost function ?? 

    
    def __init__(self):
        """ Gets a dictionary with lists of the parameters to test:
        INPUTS:
            #INTEGRATION IN THE SAME DICTIONARY THE METHOD??
            * params: dictionary with the parameters needed  f.i. {gamma:[0.1,0.4,0.9]}
            * methods: a string with the name of the method
        -------
        Author: Carles
        Date: 27/10/2013
        ---Last modification:
        Author: Antonio
        Date: 29/10/2013
        """
        self.nfolds = 1
        
    def init_clf(self):
        try:
            self.clf = self.clf_class()
            for param in self.params:
                setattr(self.clf, param,self.clf_params[param])
        
        except AttributeError:
            raise AttributeError("Error no clf_class found, is it defined at the parameter array?")
        
    
    def train_test(self,X,y):  # Change name? We train the model also.
        """
        Trains the model with the train data, test the model and return the
        performance achieved. Implicitly is making CV.
        -------
        Author: Carles
        Date: 27/10/2013
        ---Last modification:
        Author: Antonio
        Date: 29/10/2013
        """
        
        self.init_clf()
            
        #WARNING
##        Xr =  X.csr_matrix_trans()

        #nfolds = self.modelparams['folds'][0]
        Xr = csr_matrix(X)
        kfold = KFold(Xr.get_shape()[0],self.nfolds)
        

        #initialization
        acc = 0
        i = 0
        
        
        
        for train, test in kfold:
            i += 1
            Xtrain, Xtest, ytrain, ytest = Xr[train], Xr[test], y.matrix[train], y.matrix[test] #Warning: depend on X

            self.clf.fit(Xtrain, ytrain)
            ypred = self.clf.predict(Xtest)
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score

        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????
        bestclf = self.clf
        bestclf.fit(Xr,y.matrix)
	
        return  bestclf, acc #{clf: acc}


    def predict(self,test):
        ''' Make the prediction of the text using the parameters trained.
        -------
        Author: Antonio
        Date: 29/10/2013
        ---Last modification:
        Author: 
        Date: 
        '''
        return self.clf.predict(test.matrix)

    def __str__(self):
        return self.__class__.__name__ + str(self.params)

 
class TripleModel(object):  
    def __init__(self):
        """ Gets a dictionary with lists of the parameters to test:
        INPUTS:
            #INTEGRATION IN THE SAME DICTIONARY THE METHOD??
            * params: dictionary with the parameters needed  f.i. {gamma:[0.1,0.4,0.9]}
            * methods: a string with the name of the method
        -------
        Author: Carles
        Date: 27/10/2013
        ---Last modification:
        Author: Antonio
        Date: 29/10/2013
        """
        self.nfolds = 1
    
    
    
    
    def init_clfs(self):
        try:
            self.s_clf = self.s_clf_class()
            for param in self.s_params:
                setattr(self.s_clf, param,self.s_params[param])
            
            self.k_clf = self.k_clf_class()
            for param in self.k_params:
                setattr(self.k_clf, param,self.k_params[param])
            
            self.w_clf = self.w_clf_class()
            for param in self.w_params:
                setattr(self.w_clf, param,self.w_params[param])
                
        except AttributeError:
            raise AttributeError("Error no clf_class found, is it defined at the parameter array?")

    def train_test(self,X,y):  # Change name? We train the model also.
        """
        Trains the model with the train data, test the model and return the
        performance achieved. Implicitly is making CV.
        -------
        Author: Carles
        Date: 27/10/2013
        ---Last modification:
        Author: Antonio
        Date: 29/10/2013
        """
        self.init_clfs()
        

        #nfolds = self.modelparams['folds'][0]
        Xr = csr_matrix(X)
        kfold = KFold(Xr.get_shape()[0],self.nfolds)
    

        #initialization
        acc = 0
        i = 0
    
    
    
        for train, test in kfold:
            i += 1
            Xtrain, Xtest = Xr[train], Xr[test] #Warning: depend on X
            print type(y),y
            ystrain, ystest = y["s"][train],y["s"][test]
            ywtrain, ywtest = y["w"][train],y["w"][test]
            yktrain, yktest = y["k"][train],y["k"][test]
            ytest = y.matrix[test]
            
            self.s_clf.fit(Xtrain, ystrain) 
            self.k_clf.fit(Xtrain, ywtrain) 
            self.w_clf.fit(Xtrain, yktrain) 
            
            yspred = self.s_clf.predict(Xtest)
            ywpred = self.k_clf.predict(Xtest)
            ykpred = self.w_clf.predict(Xtest)
            ypred = np.concatenate((yspred,ywpred,ykpred),axis=1)
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score

        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????
        self.s_clf.fit(Xr, y["s"]) 
        self.k_clf.fit(Xr, y["k"]) 
        self.w_clf.fit(Xr, y["w"])

        return  self, acc #{clf: acc}


    def predict(self,test):
        ''' Make the prediction of the text using the parameters trained.
        -------
        Author: Antonio
        Date: 29/10/2013
        ---Last modification:
        Author: 
        Date: 
        '''
        yspred = self.s_clf.predict(test)
        ywpred = self.k_clf.predict(test)
        ykpred = self.w_clf.predict(test)
        return np.concatenate((yspred,ywpred,ykpred),axis=1)

    def __str__(self):
        return self.__class__.__name__ + str(self.s_params) + str(self.w_params) + str(self.k_params)
 
