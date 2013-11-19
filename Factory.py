
from Model import *
from Vectorizers import *
import itertools

class Factory:
    ''' Class that generalizes the GANGRENAAAA mehod.
    -------
    Author: Antonio
    Date: 29/10/2013
    ---Last modification:
    Author:
    Date:
    '''

    def create_models(self, model, params):
        models = []

        try:
            combinations = list(itertools.product(*params.values()))
        except TypeError:
            msg = "Not iterable object found while computing combinations. Are you shure that if there is a " \
                  "single value in the paramenters it is into parenthesis? "
            raise AttributeError(msg+str(params.values()))

        for c in combinations:
            curr_params = dict(zip(params.keys(),c))
            inst = SimpleModel(model, curr_params)
            models.append(inst)
   
        return models

    def create_vects(self, vect, params):
        vects = []

        try:
            combinations = list(itertools.product(*params.values()))
        except TypeError:
            msg = "Not iterable object found while computing combinations. Are you shure that if there is a " \
                  "single value in the paramenters it is into parenthesis? "
            raise AttributeError(msg+str(params.values()))

        for c in combinations:
            curr_params = dict(zip(params.keys(),c ))
            inst = SimpleVectorizer(vect, curr_params)
            vects.append(inst)

        return vects

    def create(self, meta, params):
        models = []
        try:
            combinations = list(itertools.product(*params.values()))
        except TypeError:
            msg = "Not iterable object found while computing combinations. Are you shure that if there is a " \
                  "single value in the paramenters it is into parenthesis? "
            raise AttributeError(msg+str(params.values()))

        for c in combinations:
            curr_params = dict(zip(params.keys(), c))
            inst = meta()
            for param in curr_params:
                setattr(inst, param, curr_params[param])
            models.append(inst)

        return models





