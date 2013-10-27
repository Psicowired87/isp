import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn import cross_validation
#from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix, hstack

import time

from trainer import *


# file in which we will store all the function??
from functions import *
from ModelTester import *



load_data()
X,y,test = feature_extraction()

ModelTester.search(   ) ##???
