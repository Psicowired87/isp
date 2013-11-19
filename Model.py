
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
        self.folds = None

        self.clf = model()
        for param in self.params:
            setattr(self.clf, param,self.params[param])


    def train_test(self,X, lbls):
        return self.train_test_raw(X,lbls.matrix)

    def train_test_raw(self, X, y):  # Change name? We train the model also.
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
        Xr = csr_matrix(X)
        if self.folds is None:
            self.folds = KFold(Xr.get_shape()[0], self.nfolds)

        #initialization
        acc = 0
        i = 0
        for train, test in self.folds:
            i += 1
            Xtrain, Xtest, ytrain, ytest = Xr[train], Xr[test], y[train], y[test] #Warning: depend on X
            print y.shape, ytest.shape, ytrain.shape
            self.clf.fit(Xtrain, ytrain)
            ypred = self.clf.predict(Xtest)
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score

        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????
        self.clf.fit(Xr, y)

        return self, acc


    def fit(self, Xr, y):
        self.clf.fit(Xr, y)

    def predict(self, test):
        ''' Make the prediction of the text using the parameters trained.
        -------
        Author: Antonio
        Date: 29/10/2013
        ---Last modification:
        Author: 
        Date: 
        '''
        return self.clf.predict(test)

    def __repr__(self):
        return "<SimpleModel:"+str(self.clf)+":"+str(self.params)+">"


class BlockModel(object):
    
    def __init__(self, s, w, k):
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
        self.clf_s = None
        self.clf_k = None
        self.clf_w = None
        self.results = {}
        self.folds = None

        self.nfolds = 2

    def get_models_block(self,block):
        if block == "s":
            return self.s
        elif block == "w":
            return self.w
        elif block == "k":
            return self.k
        else:
            raise AttributeError("Unknown block")

    def train_test(self, X, lbls):  # Change name? We train the model also.
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
        if self.folds is None:
            self.folds = KFold(Xr.get_shape()[0],self.nfolds)
    
        self.clf_s = self.get_best(X, lbls, "s", self.folds)
        self.clf_w = self.get_best(X, lbls, "w", self.folds)
        self.clf_k = self.get_best(X, lbls, "k", self.folds)
        
        #initialization
        acc = 0
        i = 0

       #TODO Think if this is needed
        for train, test in self.folds:
            i += 1
            Xtrain, Xtest = Xr[train], Xr[test] #Warning: depend on X
            
            ystrain, ystest = lbls["s"][train], lbls["s"][test]
            ywtrain, ywtest = lbls["w"][train], lbls["w"][test]
            yktrain, yktest = lbls["k"][train], lbls["k"][test]
            ytest = lbls.matrix[test]
            print type(ystrain)
            
            self.clf_s.fit(Xtrain, ystrain) 
            self.clf_k.fit(Xtrain, ywtrain) 
            self.clf_w.fit(Xtrain, yktrain) 
            
            yspred = self.clf_s.predict(Xtest)
            ywpred = self.clf_w.predict(Xtest)
            ykpred = self.clf_k.predict(Xtest)
            ypred = np.concatenate((yspred, ywpred, ykpred), axis=1)
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score

        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????
        self.clf_s.fit(Xr, lbls["s"])
        self.clf_k.fit(Xr, lbls["k"])
        self.clf_w.fit(Xr, lbls["w"])

        return self, acc #{clf: acc}

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
        return np.concatenate((yspred, ywpred, ykpred), axis=1)

    def get_best(self, X, lbls, block, folds):
        results_s = {}
        for model in self.get_models_block(block):
            print "testing ", block, model
            model.folds = folds
            clf, acc = model.train_test(X, lbls)
            results_s[clf] = acc
        self.results[block] = results_s

        return max(results_s, key=results_s.get)

    def __repr__(self):
        return "<BlockModel:"+str(self.clf_s)+", "+str(self.clf_w)+", "+str(self.clf_k)+">"
        

class MultiModel(object):

    def __init__(self, models):
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
        self.folds = None
        self.models = models

        self.clfs = []

    def fit(self, Xr, y):  # Change name? We train the model also.
        num_clf = y.shape[1]
        self.init_clfs(num_clf)

        self.fit_clfs(Xr, y)

    def train_test(self, X, lbls):  # Change name? We train the model also.
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
        Xr = csr_matrix(X)
        if self.folds is None:
            self.folds = KFold(Xr.get_shape()[0], self.nfolds)

        #initialization
        acc = 0
        i = 0
        for train, test in self.folds:
            i += 1
            Xtrain, Xtest, ytrain, ytest = Xr[train], Xr[test], lbls.matrix[train], lbls.matrix[test] #Warning: depend on X

            self.fit(Xtrain, ytrain)
            ypred = self.predict(Xtest)
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            print ypred.shape, ytest.shape

            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score

        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????
        self.fit(Xr, lbls.matrix)

        return self, acc

    def predict(self, test):
        ''' Make the prediction of the text using the parameters trained.
        -------
        Author: Antonio
        Date: 29/10/2013
        ---Last modification:
        Author:
        Date:
        '''
        return self.predict_clfs(test)

    def init_clfs(self,num_clfs):
        self.clfs = []
        if len(self.models) == 1:
            le_model = self.models[0]
            self.clfs.append(le_model)
            for n in range(1, num_clfs):
                new_model = SimpleModel(le_model.model, le_model.params)
                self.clfs.append(new_model)

        else:
            self.clfs = self.models

    def fit_clfs(self,X,y):
        for i in range(len(self.clfs)):
            self.clfs[i].fit(X, y[:,i])

    def predict_clfs(self,X):
        ypred = self.clfs[0].predict(X)[np.newaxis]
        ypred = ypred.transpose()
        for i in range(1, len(self.clfs)):
            y_curr = self.clfs[i].predict(X)[np.newaxis]
            y_curr = y_curr.transpose()
            ypred = np.concatenate((ypred, y_curr), axis=1)
        return ypred

    def __repr__(self):
        return "<MultiModel:"+str(self.clfs)+">"


class LabelModel(object):

    def __init__(self,lbl_models):
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

        self.lbl_models = lbl_models
        self.lbl_perf = {}
        self.results = {}
        self.clfs = []
        self.nfolds = 2
        self.folds = None

    def train_test(self, X, lbls):  # Change name? We train the model also.
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
        if len(self.clfs) == 0:
            self.init_clfs(X, lbls.matrix)
         #initialization
        acc = 0
        i = 0
        for train, test in self.folds:
            i += 1
            Xtrain, Xtest, ytrain, ytest = Xr[train], Xr[test], lbls.matrix[train], lbls.matrix[test] #Warning: depend on X

            self.fit(Xtrain, ytrain)
            ypred = self.predict(Xtest)
            ##score = accuracy_score(ytest, ypred)   ### WARNING
            score=np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))
            # print "  Fold #%d, accuracy=%f" % (i, score)
            acc += score

        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????
        self.fit(Xr, lbls.matrix)


        # Estimation of the accuracy
        acc /= self.nfolds
        #Output model trained with all the data ????

        return self, acc #{clf: acc}

    def get_best_lbl(self,X, y, i):
        results = {}
        y_i = y[:,i]
        models = self.lbl_models[i]
        for model in models:
            clf, acc = model.train_test_raw(X, y_i)
            results[clf] = acc

        self.lbl_perf[i] = results
        return max(results, key=results.get)

    def fit(self, X ,y):
        if len(self.clfs) == 0:
            self.init_clfs(X, y)
        for i in range(len(self.clfs)):
            self.clfs[i].fit(X, y[:,i])

    def init_clfs(self, X, y):
        Xr = csr_matrix(X)
        if self.folds is None:
            self.folds = KFold(Xr.get_shape()[0], self.nfolds)

        num_lbls = y.shape[1]
        self.clfs = [None] * num_lbls
        for i in range(num_lbls):
            clf = self.get_best_lbl(X, y, i)
            self.clfs[i] = clf

    def predict(self, test):
        ''' Make the prediction of the text using the parameters trained.
        -------
        Author: Antonio
        Date: 29/10/2013
        ---Last modification:
        Author:
        Date:
        '''
        ypred = self.clfs[0].predict(test)[np.newaxis]
        ypred = ypred.transpose()
        for i in range(1, len(self.clfs)):
            y_curr = self.clfs[i].predict(test)[np.newaxis]
            y_curr = y_curr.transpose()
            ypred = np.concatenate((ypred, y_curr), axis=1)
        return ypred

    def __str__(self):
        return "<LabelModel:"+str(self.s)+", "+str(self.w)+", "+str(self.k)+">"