
import numpy as np

def cosine_dist(x, y):
    '''compute cosine distance between two martrix x and y with sizes (n1, d), (n2, d)'''
    def normalize(x):
        '''normalize a 2d matrix along axis 1'''
        norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
        return x / norm
    x = normalize(x)
    y = normalize(y)
    return np.matmul(x, y.transpose([1, 0]))


