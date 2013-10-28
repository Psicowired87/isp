from ModelTester import FeatureSelectionMethod


class TFWTF(FeatureSelectionMethod):
    
    def extract(self,data):
        X = data['tweet']
        y = data[3:]
        
        return {"TFWTF":discriminative_measure(X,y)}
        
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
    