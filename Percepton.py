"""
    4 wzorce uczace:
    x1 =0,0 d1=0
    x2 =0,1 d2=0
    x3 =1,0 d3=0
    x4 = 1,0 d4=1

"""

import numpy as np


class Percepton(object):
    """
        x_vector - wektor sygnałów wejściowych, które posłużą uczeniu
        d_vector - wektor sygnałów wyjściowych, które posłużą uczeniu
        w_range - zakres losowania wag. np. w_range = 1 to losowane wagi są z przedziału [-1,1]
        threshold - próg. Występuje, gdy bias = False
        bias - dodatkowy sygnał, pomagający w szukaniu wag
        unipolar = True - funkcja progowa będzie unipolarna, bipolarna w przeciwnym wypadku
        alpha - współczynnik uczenia - określa jak szybko będzie zmieniała się waga
    """

    def train(self, x_vector, d_vector, w_range, threshold, unipolar, alpha, bias):

        if bias:
            for x in x_vector:
                x.insert(0, 1)
            w_vector = self.make_weights(x_vector[0], w_range)

        else:
            w_vector = self.make_weights(x_vector[0], w_range)

        # dla kazdego wzorca
        without_faults = False
        epoch = 0
        while not without_faults:

            without_faults = True
            epoch += 1
            faults = []
            for x, d in zip(x_vector, d_vector):
                w_vector_updated = []
                y = self.activate(w_vector, x, unipolar, threshold, bias)
                fault = d - y
                for w, x_sample in zip(w_vector, x):
                    w_vector_updated.append(w + (alpha * (fault * x_sample)))
                w_vector = w_vector_updated
                faults.append(fault)
            for f in faults:
                if f != 0:
                   without_faults = False
        if not bias:
            print("prog:" + str(threshold))
        self.w_vector = w_vector
        self.threshhold = threshold
        self.bias = bias
        self.unipolar = unipolar
        print("Wspolczynnik uczenia: " + str(alpha))

        print("Czy z biasem?:" + str(bias))
        print("Czy unipolarna?: " + str(unipolar))
        print("Zakres wag: [" + str(-w_range) + ", " + str(w_range) + "]")
        print("liczba epok:" + str(epoch))
        print("Ostateczne wagi: " + str(w_vector))
        # self.update_weights(self.w_vector, faults_vector, alpha)

    def make_weights(self, x, w_range):
        w_vector = np.random.randint(-(w_range * 100), w_range * 100, len(x))
        w_vector = w_vector / (np.random.randint(100, 10000))
        return np.round(w_vector, 5)

    def activate(self, w_vector, x, unipolar, threshold, bias):
        z = 0
        # z
        for i in range(len(x)):
            z += w_vector[i] * x[i]
        # unipolar
        if bias:
            threshold = 0
        if unipolar:
            if z > threshold:
                return 1
            else:
                return 0
        # bipolar
        else:
            if z > threshold:
                return 1
            else:
                return -1

    def predict(self, x_sample):
        y = self.activate(self.w_vector, x_sample, self.unipolar, self.threshhold, self.bias)
        print("Predicted y = " + str(y))
