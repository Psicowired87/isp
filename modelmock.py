__author__ = 'carles'
import numpy as np
import random


class Model:
    def predict(self, tweet):
        return np.array((random.random() for i in range(24)))