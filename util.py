from math import erf, exp, sqrt

SQRT_2 = 1.4142135623730951

def islist(l):
    return type(l) == list

def vectorize(f, X):
    return map(f, X) if islist(X) else f(X)

def cdf(X, mean=0, std=1):
    """
    Cumulative distribution function.
    """
    return vectorize(lambda x: 0.5 * (1 + erf((x - mean)/(std*SQRT_2))), X)

def scale(X, mean, sdev):
    """
    R like scale function.
    """
    return vectorize(lambda x: (x-mean)/sdev, X)

def labs(X):
    return vectorize(abs, X) 

def minus(X):
    return vectorize(lambda x: -x, X)

def zscore(X, mean, sdev):
    return vectorize(lambda x: 2*x, cdf(minus(labs(scale(X, mean, sdev)))))

def sd(X, mean):
    return sqrt(var(X, mean))

def var(X, mean):
    return sum(map(lambda x: pow(x - mean, 2), X)) / (len(X)-1)

def projection(collection, name):
    return map(lambda x: x[name], collection)

def edist(a, b):
    """
    Euclidean distance
    """
    return sqrt(pow(b[0]-a[0], 2) + pow(b[1]-a[1], 2))

def mdist(a, b):
    """
    Manhattan distance
    """
    return (abs(b[0]-a[0]) + abs(b[1] - a[1]))
        
def angle(v1, v2):
    """
    Calculate the angle between vectors.
    """
    dot_prod = np.dot(v1, v2)
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    return np.arccos(dot_prod / (m1 * m2))

