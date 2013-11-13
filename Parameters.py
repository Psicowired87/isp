import itertools


class Parameters:
    '''Class which represents all the parameters we will use.
    Store all the parameters in a centralized way and retrieve models and vectorizers.
    '''


    def __init__(self,modelparams=dict(),vectparams = dict(),searchparams = dict()):
        '''
        INPUTS:
            * trainers: list of words (models)
            * modelparams: dictionary of paramters with list of possible values
            * vectparams: dictionary of paramters with list of possible values
        '''
        self.searchparams = searchparams
        self.modelparams = modelparams
        self.vectparams = vectparams
        