__author__ = 'carles'
import numpy as np
import random


class Model:
    def predict(self, tweets):
        """
        Gets a list with n tweets and returns a np.array of 24 columns with the masses of the labels
        @param tweet: The n tweets as input
        @return:np.array with dimension n x 24
        """
        return np.array([[random.random() for i in range(24)] for j in range(len(tweets))])



class ModelException(Exception):
    pass

# lbls = ["I can't tell",
#                 "Negative",
#                 "Neutral",
#                 "Positive",
#                 "Tweet not related to weather condition",
#                 "current (same day) weather",
#                 "future (forecast)",
#                 "I can't tell",
#                 "past weather",
#                 "clouds",
#                 "cold",
#                 "dry",
#                 "hot",
#                 "humid",
#                 "hurricane",
#                 "I can't tell",
#                 "ice",
#                 "other",
#                 "rain",
#                 "snow",
#                 "storms",
#                 "sun",
#                 "tornado",
#                 "wind",]