
# Global
import numpy as np


# import measures and post-selection of features
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

#import methods
from sklearn import linear_model

# import cross-validation
from sklearn.cross_validation import KFold


# Provably temporal. May be they have to be incorporated in classes



def crossvalid(X,y,N_FOLDS,clfs,scoreFuncs):

    # Search best number of features
    N_FEATURES = 10000
    nFeatures = np.array([10000])


    # Vector of names of methods
    clfnames = map(lambda x: type(x).__name__
                   if type(x).__name__ != 'OneVsRestClassifier'
                   else type(x.estimator).__name__, clfs)

    #initialization of the accuracies
    accuracies1 = np.zeros((len(clfs), len(nFeatures), len(scoreFuncs)))
    accuracies2 = np.zeros((len(clfs), len(nFeatures), len(scoreFuncs)))
    accuracies3 = np.zeros((len(clfs), len(nFeatures), len(scoreFuncs)))


    # In parts
    y1, y2, y3 = np.array(t.ix[:,4:9]), np.array(t.ix[:,9:13]), np.array(t.ix[:,13:])



    # Bucle to call evaluate for all parameters and methods
    for k in range(0, len(scoreFuncs)):
        Xtrunc1 = X.copy()
        Xtrunc2 = X.copy()
        Xtrunc3 = X.copy()
        for j in range(0, len(nFeatures)):
            if nFeatures[j] != N_FEATURES:
                featureSelector1 = SelectKBest(score_func=scoreFuncs[k], k=nFeatures[j])
                Xtrunc1 = featureSelector1.fit_transform(X, y1)
                featureSelector2 = SelectKBest(score_func=scoreFuncs[k], k=nFeatures[j])
                Xtrunc2 = featureSelector2.fit_transform(X, y2)
                featureSelector3 = SelectKBest(score_func=scoreFuncs[k], k=nFeatures[j])
                Xtrunc3 = featureSelector3.fit_transform(X, y3)
            for i in range(0, len(clfs)):
                accuracies1[i, j, k] = evaluate(Xtrunc1, y1, N_FOLDS, clfs[i],
                                               nFeatures[j], clfnames[i], scoreFuncs[k])
                accuracies2[i, j, k] = evaluate(Xtrunc2, y2, N_FOLDS, clfs[i],
                                               nFeatures[j], clfnames[i], scoreFuncs[k])
                accuracies3[i, j, k] = evaluate(Xtrunc3, y3, N_FOLDS, clfs[i],
                                               nFeatures[j], clfnames[i], scoreFuncs[k])

    
    #i,j,k = np.unravel_index(accuracies.argmax(), accuracies.shape)
    parameters = [ np.unravel_index(accuracies1.argmax(), accuracies1.shape),
                   np.unravel_index(accuracies2.argmax(), accuracies2.shape),
                   np.unravel_index(accuracies3.argmax(), accuracies3.shape)]
    return parameters


def predicting(X,y,parameters,Xtest):
    y1, y2, y3 = np.array(t.ix[:,4:9]), np.array(t.ix[:,9:13]), np.array(t.ix[:,13:])

    #parameters[][]

    Xtrunc1 = X.copy()
    featureSelector1 = SelectKBest(score_func=scoreFuncs[param[0][2]], k=nFeatures[param[0][1]])
    Xtrunc1 = featureSelector1.fit_transform(X, y1)
    Xtrunc2 = X.copy()
    featureSelector2 = SelectKBest(score_func=scoreFuncs[param[1][2]], k=nFeatures[param[1][1]])
    Xtrunc2 = featureSelector2.fit_transform(X, y2)
    Xtrunc3 = X.copy()
    featureSelector3 = SelectKBest(score_func=scoreFuncs[param[2][2]], k=nFeatures[param[2][1]])
    Xtrunc3 = featureSelector3.fit_transform(X, y3)

    clf1 = clfs[param[0][0]]
    clf2 = clfs[param[1][0]]
    clf3 = clfs[param[2][0]]

    clf1.fit(X,y1)
    clf2.fit(X,y2)
    clf3.fit(X,y3)
    
    ypred = np.concatenate((clf1.predict(Xtest),clf2.predict(Xtest),clf3.predict(Xtest)),axis=1)
    

    return ypred
    

def evaluate(X, y, nfolds, clf, nfeats, clfname, scoreFunc):
    kfold = KFold(X.shape[0], nfolds)
    acc = 0
    i = 0
    print("%s (#-features=%d)..." % (clfname, nfeats))

    # Cross-validation
    for train, test in kfold:
        i += 1
        Xtrain, Xtest, ytrain, ytest = X[test], X[train], y[test], y[train]
        clf.fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        score = accuracy_score(ytest, ypred)
        print "  Fold #%d, accuracy=%f" % (i, score)
        acc += score
    acc /= nfolds
    print "## %s (#-features=%d) accuracy=%f" % (clfname, nfeats, acc)
    return acc


def accuracy_score(ytest,ypred):
    #print ytest.shape, ypred.shape
    size = ytest.shape
    elements = size[0]*size[1]
#    return sum(sum(np.array(ytest)!=np.array(ypred)))/elements
    #a = np.equal(ytest,ypred)
    return np.sqrt(np.sum(np.array(ypred-ytest)**2)/ (ypred.shape[0]*24.0))




def train(X,y):
    print 'Training...',
    timestamp1 = time.time()


    # SPACE PARAMETER ###### We have to search the best ones in CV
    N_FOLDS = 2
    # Vector of methods
    clfs = [
        linear_model.Ridge(alpha = 0.05),
        linear_model.LinearRegression()
      ]
    #Feature selection measures:
    #scoreFuncs = [chi2, f_classif]
    scoreFuncs = [chi2]
    ###################

    parameters = crossvalid(X,y,N_FOLDS,clfs,scoreFuncs)

    timestamp2 = time.time()
    print "%.2f seconds" % (timestamp2 - timestamp1)

    return parameters




############# PREDICTION #############
def predict(parameters, test):
    
    print 'Prediction...',
    timestamp1 = time.time()
    
    test_prediction = predicting(X,y,parameters,test)
    
    timestamp2 = time.time()
    print "%.2f seconds" % (timestamp2 - timestamp1)

    return test_prediction


############# SAVE PREDICTION RESULTS #############
def saveResults(ofname, test_prediction):
    print 'Save prediction results...',
    timestamp1 = time.time()

    aHeader = 'id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15'

    #prediction = np.array(np.hstack([np.matrix(t2['id']).T, new_array]))  
    prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))
    col = '%i,' + '%f,'*23 + '%f'
    np.savetxt(ofname, prediction,col, delimiter=',', header=aHeader)

    timestamp2 = time.time()
    print "%.2f seconds" % (timestamp2 - timestamp1)





# May be define a class for X with this function??
def discriminative_measure(X,y):
    ''' It is a measure of discriminance for multiclass problem.
    This measure have some properties:
    - It is scaled measure:
        * discriminative_measure(priors) = 0
        * discriminative_measure(extreme_case)=1    ;e.g. extreme_case=[1,0,0,0,0]

    -------------------------------------
    INPUTS:
        * X: vectorized representation of features <class 'scipy.sparse.csr.csr_matrix'>
        * y: multiclass tags
    OUTPUT:
        * measure of discriminance for every feature in order.
    -------------------------------------
    +Info: Toño
    '''
    
    #It computes the average value of every feature for the vector of tags (distribution)
    index_matrix = [X[i].indices for i in range(X.shape[1]) ]
    distribution = [sum(y[index_matrix[i]])/len(index_matrix[i]) for i in range(len(index_matrix))]
    # the priors is the average of tags of all the features. It is the definition of a non-discriminative feature
    priors = sum(y)/len(y)
    # We compute the average scaled distance to the priors:
    distance_to_priors = np.array([np.array(distribution[i]-priors) for i in range(len(distribution)) ])
    discriminative = (distance_to_priors>0)*(distance_to_priors/(1-priors)) - (distance_to_priors<0)*(distance_to_priors/(priors))
    return np.sum(discriminative,axis=1)/np.array(y).shape[1]




