import numpy as np

def generate_matrix(n):
    matrix = np.random.randint(100000, size=(n, n))
    filename = 'matrix{}.txt'.format(n)
    np.savetxt(filename, matrix, fmt='%d')


generate_matrix(2**14)

