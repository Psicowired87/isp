
# Interns
from load_data_module import *
from sklearn.linear_model import *
from sklearn.feature_extraction.text import CountVectorizer
from Parameters import Parameters
from Labels import Labels
from Factory import *
from ModelTester import ModelTester
from Vectorizers import *
from Model import *
from sklearn.svm import *
from sklearn.neighbors import KNeighborsClassifier
##from Vectorizer import *


def test_factory():
    factory = Factory()
    models = factory.create_models(Ridge,{'alpha':[0.4,0.5]})
    assert models[0].clf.alpha == 0.4 and models[1].clf.alpha

def test_factory2():
    factory = Factory()
    models = factory.create_models(Ridge,{'alpha':[0.4,0.5], 'fit_intercept':[False,True]})
    assert models[0].clf.alpha == 0.4 and models[0].clf.fit_intercept == False and models[1].clf.alpha == 0.4 and \
                                    models[1].clf.fit_intercept == True and len(models) == 4

def test_simple(X, y):
    #Test simplest case
    simple = SimpleModel(Ridge, {"alpha":0.4})
    vect = SimpleVectorizer(CountVectorizer,{})
    tester = ModelTester()
    results = tester.search([vect], [simple], X, y)
    assert len(results) == 1


def test_multimodel(X,y):
    vect = CountVectorizer()
    m = [SimpleModel(Ridge, {})] #Or num_labels models
    multi = MultiModel(m)
    X = vect.fit_transform(X,y.matrix)
    res = multi.train_test(X,y)
    print res
    assert res is not None


def test_block(X,y):
    model_class1, model_class2 = Ridge, LinearRegression
    parameters1, parameters2 = {"alpha": 0.3}, {}
    s = [SimpleModel(model_class1, parameters1), SimpleModel(model_class2, parameters2)]
    k = [SimpleModel(model_class1, parameters1), SimpleModel(model_class2, parameters2)]
    w = [SimpleModel(model_class1, parameters1), SimpleModel(model_class2, parameters2)]
    vect = CountVectorizer()
    triple = BlockModel(s, k, w)
    X = vect.fit_transform(X,y.matrix)
    res = triple.train_test(X,y)
    assert res is not None

def test_label(X,y):
    l = []
    for i in range(24): l.append([SimpleModel(Ridge, {}), SimpleModel(SVR, {})])
    lbl = LabelModel(l)
    vect = CountVectorizer()
    X = vect.fit_transform(X,y.matrix)
    res = lbl.train_test(X,y)
    assert res is not None

def test_search(X, y):
    #Test simplest case
    factory = Factory()
    models = factory.create_models(Ridge, {'alpha': [0.3, 0.5, 0.7], 'fit_intercept': [False, True]})
    #TODO put another model that WORKS
    vects = factory.create_vects(CountVectorizer, {'ngram_range': [(1, 2), (1, 3)], 'max_features': [None, 1000, 100000]})
    tester = ModelTester()
    results = tester.search(vects, models, X, y)
    print len(results)
    assert len(results) == 36 #3*2 models * 3*2 vects = 36

# load train as t and test as t2
t,t2 = load_data('./train.csv', './test.csv')

#TODO remove limits only for test
y = Labels(np.array(t.ix[:9, 4:]))
X = np.array(t['tweet'])[:10]


#test_simple(X,y)
#test_factory()
#test_factory2()
#test_multimodel(X,y)
#test_block(X,y)
#test_label(X,y)
test_search(X,y)

"""
model_class, model_class1, model_class2= Ridge, Ridge, LinearRegression
parameters, parameters1, parameters2 ={"alpha": [1, 0.2]}, {"alpha":[0.4, 0.3],"fit":[True,False]}, {}
#Simple model that does no testing
simple = SimpleModel(model_class, parameters)

#SimpleModel factory
factory = Factory()

models = factory.create_models(model_class, parameters)

#Test for models with different models for each block, to find
# to find the best among them, then use the best combination
# as the simple

s = [SimpleModel(model_class1, parameters1), SimpleModel(model_class2, parameters2)]
k = [SimpleModel(model_class1, parameters1), SimpleModel(model_class2, parameters2)]
w = [SimpleModel(model_class1, parameters1), SimpleModel(model_class2, parameters2)]

triple = BlockModel(s, k, w)

#Use non-multiclass models, such as SVR, into the same way than SimpleModel
m = [SimpleModel(model_class, parameters)] #Or num_labels models
multi = MultiModel(m)

models.extend([multi, triple])
models = [simple]
#Test for models with different models for each label in the same way than triple model
# and pick the best
l = []
for i in range(24): l.append([SimpleModel(model_class, parameters)])

lbl = LabelModel(l)

tester = ModelTester()

vect_class = CountVectorizer
vectorizers = factory.create_vects(vect_class, parameters)

results = tester.search(vectorizers, models, X, y)
"""