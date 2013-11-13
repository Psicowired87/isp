# Module to load and preprocess data
import numpy as np
import pandas as p
import time

def load_data(trainFile='train.csv', testFile='test.csv'):
    ''' Function load_data: loads data from train and test files
        INPUT:
            * trainFile: file name of training. Default: 'train.csv'
            * testFile: file name of test. Default: 'test.csv'
        OUTPUT:
            * t: train as <class 'pandas.core.frame.DataFrame'>
            * t2: test as <class 'pandas.core.frame.DataFrame'>
    AUTHOR: Oualid, Antonio
    DATE: 29/10/2013
    '''   
    print 'Loading data...',
    timestamp1 = time.time()

    t = p.read_csv(trainFile)
    t2 = p.read_csv(testFile)
	
    timestamp2 = time.time()
    print "%.2f seconds" % (timestamp2 - timestamp1)

    return t, t2
    
############# SAVE PREDICTION RESULTS #############
def saveResults(ofname, test_prediction,t2):
    print 'Save prediction results...',
    timestamp1 = time.time()

    aHeader = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"

    #prediction = np.array(np.hstack([np.matrix(t2['id']).T, new_array]))  
    prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))
    col = '%i,' + '%f,'*23 + '%f'
    np.savetxt('tmp.csv', prediction,col, delimiter=',')
    f = open(ofname,'w')
    f.write(aHeader+'\n')
    n = open('tmp.csv')
    for line in n:
        f.write(line)
        
    f.close()
    n.close()
    timestamp2 = time.time()
    print "%.2f seconds" % (timestamp2 - timestamp1)



##Future work:
##    1. Preprocessing data:
##    * Remove @mention
##    * Remove {link}
##    * Remove or separate #hashtag
##    * Add complementary tags describing emoticons
##    * Language detection and translation
##
