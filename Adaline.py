import numpy as np


class Adaline:
    """
    logic = "AND" "OR"
    activation_function: True = unipolar / False = bipolar
    Threshold = 0 -> we use bias instead
    delta = learning rate
    permissible_error - dopuszczalny blad
    """

    def __init__(self, weight_range=1, delta=0.01, permissible_error=0.5, max_epochs=10000, activation_function=False):
        self.__weight_range = weight_range
        self.__max_epochs = max_epochs
        self.__delta = delta
        self.__activation_function = activation_function
        self.__permissible_error = permissible_error

        self.__x_pattern = None
        self.__y_pattern = None
        self.__w_pattern = None

    def train(self):
        """

        """
        self.__prepareTrainingData()
        epoch = -1
        epoch_error = 10
        permissible_error = self.__permissible_error

        #dla unipolarnej
        while epoch_error >= permissible_error and epoch < self.__max_epochs:
            errors_in_epoch = []
            epoch += 1
            for x, d in zip(self.__x_pattern, self.__y_pattern):
                updated_w = []
                error = d - self.__alc(x)
                for w, single_x in zip(self.__w_pattern, x):
                    updated_w.append(w + (2 * self.__delta * error * single_x))
                self.__w_pattern = updated_w
                errors_in_epoch.append(error)
            epoch_error = sum(np.power(errors_in_epoch, 2))/len(errors_in_epoch)
        self.__w_pattern = np.round(self.__w_pattern, 3)
        self.__showAdalineDetails(epoch, epoch_error)

    def __prepareTrainingData(self):
        """
        Przygotowuje wszystkie dane dla uczenia
        """
        self.__prepareDefaultTrainPattern()
        self.__addBiasToPatterns()
        self.__generateTrainingWeights()

    def __prepareDefaultTrainPattern(self):
        """
        Przygotowuje wzorce uczące w zależnosci, czy będziemy stosować funkcje aktywacji bipolarna czy unipolarna
        """
        if self.__activation_function:
            self.__x_pattern = [[0, 0], [0, 1], [1, 0], [1, 1]]
            self.__y_pattern = [0, 0, 0, 1]
        else:
            self.__x_pattern = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
            self.__y_pattern = [-1, -1, -1, 1]

    def __addBiasToPatterns(self):
        """
        Dodaje bias
        """
        for x in self.__x_pattern:
            x.insert(0, 1)

    def __generateTrainingWeights(self):
        """
        Przygotowuje wagi w zależności od wejść\n
        input_pattern_len = długość jednego wzorca sygnałów wejściowych, aby można było skonstruować wektory wag
        """
        input_pattern_len = len(self.__x_pattern[0])
        self.__w_pattern = np.random.randint(-(self.__weight_range * 100), self.__weight_range * 100, input_pattern_len)
        self.__w_pattern = self.__w_pattern / (np.random.randint(100, 10000))
        self.__w_pattern = np.round(self.__w_pattern, 3)

    def __alc(self, x_sample):
        z = 0
        for i in range(len(x_sample)):
            z += self.__w_pattern[i] * x_sample[i]
        return z

    def __showAdalineDetails(self, epoch, epoch_error):
        """
                Wypisuje szczegoly uczenia
                """
        print("\nWłaściwości uczenia:")
        print("liczba epok:" + str(epoch))
        print("Wspolczynnik uczenia: " + str(self.__delta))
        print("Zakres wag: [" + str(-self.__weight_range) + ", " + str(self.__weight_range) + "]")
        print("Dopuszczalny błąd: " + str(self.__permissible_error))
        print("Otrzymany błąd:" + str(epoch_error))
        print("Ostateczne wagi: " + str(self.__w_pattern))
        if self.__activation_function:
            print("Funkcja aktywacji = unipolarna")
        else:
            print("Funkcja aktywacji = bipolarna")

    def predict(self, x):
        z = self.__alc(x)
        if z > 0:
            return 1
        elif self.__activation_function:
            return 0
        else:
            return -1
