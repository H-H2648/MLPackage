import numpy as np


class BasisExpansion:
    #inputArray is of the form: 
    # assumes input is of the form [[x11, x12, x13, ..., x1p], [x21, x22, ..., x2p], [xN1, xN2, ..., xNp]] (numpy array)
    # assumes output is of the form [y1 | y2| y3|, ... , |yK] (numpy array)
    #where yij = 1 iff row j corresponds to the classification j
    #indicatorArray is of the form [G1, G2, ..., GK] where Gi corresponds to classification, i
    # propered implies that someone has already turned the input to be of the form [[1, x11, x12, x13, ..., x1p], [1, x21, x22, ..., x2p], [1, xN1, xN2, ..., xNp]]
    # normalized implies that someone has already turned the input to be of the form [[1, x11, x12, x13, ..., x1p], [1, x21, x22, ..., x2p], [1, xN1, xN2, ..., xNp]]
    def __init__(self):
        pass
    
    def getKnots(self, n, predictors):
        minKnot = np.amin(predictors)
        maxKnot = np.amax(predictors)
        gap = maxKnot - minKnot
        knots = np.zeros(n)
        for ii in range(n):
            knots[ii] = minKnot + gap*(ii)/(n-1)
        return knots

    def piecewiseLinear(self, x, knot):
        return np.where(x > knot, x - knot, 0)
    
    def cubeKnot(self, x, knots, ii):
        currentKnot = knots[ii]
        maxKnot = knots[-1]
        return (self.piecewiseLinear(x, currentKnot)**3 - self.piecewiseLinear(x, maxKnot)**3)/(maxKnot - currentKnot)

    def naturalCubeSplines(self, x, knots, ii):
        return self.cubeKnot(x, knots, ii) - self.cubeKnot(x, knots, -2)





    


        
        

            


