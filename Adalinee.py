import numpy as np


class Adalinee:
    """
    logic = "AND" "OR"
    activation_function: True = unipolar / False = bipolar
    Threshold = 0 -> we use bias instead
    """

    def __init__(self, weight_range=1, delta=0.01, activation_function=True):
        self.__weight_range = weight_range
        self.__delta = delta
        self.__activation_function = activation_function

        self.__x_pattern = None
        self.__y_pattern = None
        self.__w_pattern = None

