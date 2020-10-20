import numpy as np


class Percepton:
    """
    logic = "AND" "OR"
    activation_function: True = unipolar / False = bipolar
    Threshold = 0 -> we use bias instead
    """

    def __init__(self, weight_range=1, alpha=0.01, threshold=0, activation_function=True):
        self.__weight_range = weight_range
        self.__alpha = alpha
        self.__threshold = threshold
        if self.__threshold == 0:
            self.__bias = True
        else:
            self.__bias = False
        self.__activation_function = activation_function

        self.__x_pattern = None
        self.__y_pattern = None
        self.__w_pattern = None

    def train(self):
        """
        epoch - liczba zawierajaca epoki\n
        trained - jeśli jest ustawiony na true, wtedy oznacza to, ze znaleziono wagi, przy ktorych blad = 0\n
        x - pojedynczy wzorzec\n
        d - pozadany sygnal wyjsciowy dla wzorca x\n
        y - otrzymany sygnal wyjsciowy przy pomocy funkcji aktywacji\n
        error - otrzymany blad dla wzorca\n
        w - pojedyncza waga\n
        single_x - pojedynczy sygnal wzorca\n
        error_in_epoch - tablica przechowujaca bledy z kazdej epoki\n
        updated_w - tablica przechowujaca uaktualnione wagi
        """
        self.__prepareTrainingData()
        epoch = 0
        trained = False
        while not trained:
            trained = True
            epoch += 1
            errors_in_epoch = []
            for x, d in zip(self.__x_pattern, self.__y_pattern):
                updated_w = []
                y = self.__activate(x)
                error = d - y
                for w, single_x in zip(self.__w_pattern, x):
                    updated_w.append(w+(self.__alpha*(error*single_x)))
                self.__w_pattern = updated_w
                errors_in_epoch.append(error)
            for e in errors_in_epoch:
                if e != 0:
                    trained = False
        self.__w_pattern = np.round(self.__w_pattern, 3)
        self.__showPerceptonDetails(epoch)

    def __prepareTrainingData(self):
        """
        Przygotowuje wszystkie dane dla uczenia
        """
        self.__prepareDefaultTrainPattern()
        if self.__bias:
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

    def __activate(self, x_sample):
        """
        Funkcja aktywacji dla pojedyńczego wzorca
        z - całkowite pobudzenie neuronu
        x_sample - pojedyńczy wzorzec wejściowy
        """
        z = 0
        for i in range(len(x_sample)):
            z += self.__w_pattern[i] * x_sample[i]
        if z > self.__threshold:
            return 1
        elif self.__activation_function:
            return 0
        else:
            return -1

    def __showPerceptonDetails(self, epoch):
        """
        Wypisuje szczegoly uczenia
        """
        print("\nWłaściwości uczenia:")
        print("liczba epok:" + str(epoch))
        print("Wspolczynnik uczenia: " + str(self.__alpha))
        if self.__bias:
            print("Wykorzystano bias")
        else:
            print("Próg: " + str(self.__threshold))
        if self.__activation_function:
            print("Funkcja aktywacji - unipolarna")
        else:
            print("Funkcja aktywacji - bipolarna")
        print("Zakres wag: [" + str(-self.__weight_range) + ", " + str(self.__weight_range) + "]")
        print("Ostateczne wagi: " + str(self.__w_pattern))
