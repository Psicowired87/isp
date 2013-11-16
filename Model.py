
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
from scipy.sparse import csr_matrix
import itertools




from sklearn.cross_validation import KFold
import numpy as np



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

    
    def __init__(self, model, params):
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
        self.nfolds = 2
        self.model = model
        self.params = params
        
        print model
        self.clf = model()
        for param in self.params:
            setattr(self.clf, param,self.params[param])
        
        
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


    def predict(self, test):
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

    def __str__(self):
        return "<SimpleModel:"+str(self.clf)+">"


class TripleModelSearch(object):  
    
    def __init__(self,s,w,k):
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
        
        self.s = s
        self.w = w
        self.k = k
        
        self.nfolds = 2
    
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

        #nfolds = self.modelparams['folds'][0]
        Xr = csr_matrix(X)
        kfold = KFold(Xr.get_shape()[0],self.nfolds)
    
        self.clf_s = self.get_best_s(Xr,y,kfold)
        self.clf_w = self.get_best_w(Xr,y,kfold)
        self.clf_k = self.get_best_k(Xr,y,kfold)
        
        #initialization
        acc = 0
        i = 0
        
        for train, test in kfold:
            i += 1
            Xtrain, Xtest = Xr[train], Xr[test] #Warning: depend on X
            
            ystrain, ystest = y["s"][train],y["s"][test]
            ywtrain, ywtest = y["w"][train],y["w"][test]
            yktrain, yktest = y["k"][train],y["k"][test]
            ytest = y.matrix[test]
            
            self.clf_s.fit(Xtrain, ystrain) 
            self.clf_k.fit(Xtrain, ywtrain) 
            self.clf_w.fit(Xtrain, yktrain) 
            
            yspred = self.clf_s.predict(Xtest)
            ywpred = self.clf_w.predict(Xtest)
            ykpred = self.clf_k.predict(Xtest)
            ypred = np.concatenate((yspred,ywpred,ykpred),axis=1)
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score

        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????
        self.clf_s.fit(Xr, y["s"]) 
        self.clf_k.fit(Xr, y["k"]) 
        self.clf_w.fit(Xr, y["w"])

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
        yspred = self.clf_s.predict(test)
        ywpred = self.clf_k.predict(test)
        ykpred = self.clf_w.predict(test)
        return np.concatenate((yspred,ywpred,ykpred),axis=1)


 
    def get_best_s(self,X,y,kfold):
        
        self.results_s = {}
        models = self.get_all_combinations(self.s)
        if len(models) == 1:
            return models[0]
        for model in models:
            print "testing S",model
            acc = self.do_cv(X,y,kfold,model,"s")
            self.results_s[model] = acc
            
        return max(self.results_s, key=self.results_s.get)
    
        
    def get_best_w(self,X,y,kfold):
        
        self.results_w = {}
        models = self.get_all_combinations(self.w)
        if len(models) == 1:
            return models[0]
        for model in models:
            print "testing w", model
            acc = self.do_cv(X,y,kfold,model,"w")
            self.results_w[model] = acc
            
        return max(self.results_w, key=self.results_w.get)
    
    def get_best_k(self, X, y,kfold):
        
        self.results_k = {}
        models = self.get_all_combinations(self.k)
        if len(models) == 1:
            return models[0]
        for model in models:
            print "testing k", model
            acc = self.do_cv(X,y,kfold,model,"k")
            self.results_k[model] = acc
            
        return max(self.results_k, key=self.results_k.get)
        
        
    def get_all_combinations(self,test):
        
        if len(test) == 0:
            raise AttributeError("Empty model test list")
            
        models = []
        try:
            for model in test:
                a = [vals for vals in test[model].values()]           
                combinations = list(itertools.product(*a))
            
                for c in combinations:
                    curr_params = dict(zip(test[model].keys(),c))
                
                    try:
                        clf = model()

                        for param in curr_params:
                            print param
                            setattr(clf, param, curr_params[param])
                        models.append(clf)
                    except AttributeError:
                        raise AttributeError("Error no clf "+str(model)+" found, is it defined at the parameter array?")
                        
        except TypeError:
            raise AttributeError("""Not iterable object found while computing combinations. 
            Are you shure that if there is a single value in the paramenters it is into parenthesis?""")
        return models
        
    def do_cv(self, Xr, y, kfold, clf, block):
        #initialization
        acc = 0
        i = 0
        
        for train, test in kfold:
            i += 1
            Xtrain, Xtest = Xr[train], Xr[test] #Warning: depend on X
            
            ytrain, ytest = y[block][train],y[block][test]
            

            
            clf.fit(Xtrain, ytrain)
            
            ypred = clf.predict(Xtest)
            
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score
            print "F",i
        # Estimation of the accuracy
        acc /= self.nfolds
        return acc

    def __str__(self):
        return "<TripleModel:"+str(self.s)+", "+str(self.w)+", "+str(self.k)+">"
        
    
    
    