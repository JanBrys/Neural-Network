import numpy as np


class Adaline(object):

    def train(self, x_vector, d_vector, w_range, learning_rate, permissible_error):
        for x in x_vector:
            x.insert(0, 1)
        w_vector = self.make_weights(x_vector[0], w_range)
        final_error = 10000000000
        epoch = 0
        run = True
        while epoch < 10 or run == True:
            faults = []
            run = False
            epoch += 1
            for x, d in zip(x_vector, d_vector):
                w_vector_updated = []
                error = d - self.alc(w_vector, x)
                for w, x_sample in zip(w_vector, x):
                    w_vector_updated.append(w + (2 * learning_rate * error * x_sample))
                w_vector = w_vector_updated
                faults.append(error)
            final_error = sum(faults) / (len(x_vector))
            print(final_error)
            if final_error >= permissible_error:
                run = True

        self.w_vector = w_vector
        print("liczba epok:" + str(epoch))
        print("Ostateczne wagi: " + str(w_vector))
        print("Ostateczny błąd: " + str(final_error))

    def make_weights(self, x, w_range):
        w_vector = np.random.randint(-(w_range * 100), w_range * 100, len(x))
        w_vector = w_vector / (np.random.randint(100, 10000))
        return np.round(w_vector, 5)

    # zastosowac bipolarny
    def alc(self, w_vector, x):
        z = 0
        for i in range(len(x)):
            z += w_vector[i] * x[i]
        return z
