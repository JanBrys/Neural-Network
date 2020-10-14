from Adaline import Adaline
from Percepton import Percepton

if __name__ == "__main__":
    x_vector = [[0, 0], [0, 1], [1, 0], [1, 1]]    # sygnały wejściowe uczące
    d_vector = [0, 0, 0, 1]                       #przewidywane wyjscia

    p1 = Percepton()
    p1.train(x_vector,d_vector,1,0.5,True,0.01,True)

    #p1.predict([0.99, 0.98])

    #a1 = Adaline()
    #a1.train(x_vector,d_vector, 0.8, 0.01, 0.001)