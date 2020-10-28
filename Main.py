from Percepton import Percepton
from Adaline import Adaline

if __name__ == "__main__":
    x_vector = [[0, 0], [0, 1], [1, 0], [1, 1]]  # sygnały wejściowe uczące
    d_vector = [0, 0, 0, 1]  # przewidywane wyjscia

    p1 = Percepton()
    p1.task1(weight_range=1.0, bias=True, activation_function=True)

    #a1 = Adaline()
    #a1.task1(delta=0.0015, permissible_error=0.266)
