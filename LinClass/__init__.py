import numpy as np
import math
from scipy import linalg


class LinClass:
    #inputArray is of the form: 
    # assumes input is of the form [[x11, x12, x13, ..., x1p], [x21, x22, ..., x2p], [xN1, xN2, ..., xNp]] (numpy array)
    # assumes output is of the form [y1 | y2| y3|, ... , |yK] (numpy array)
    #where yij = 1 iff row j corresponds to the classification j
    #indicatorArray is of the form [G1, G2, ..., GK] where Gi corresponds to classification, i
    # propered implies that someone has already turned the input to be of the form [[1, x11, x12, x13, ..., x1p], [1, x21, x22, ..., x2p], [1, xN1, xN2, ..., xNp]]
    # normalized implies that someone has already turned the input to be of the form [[1, x11, x12, x13, ..., x1p], [1, x21, x22, ..., x2p], [1, xN1, xN2, ..., xNp]]
    def __init__(self, inputArray, outputArray, indicatorArray, normalized=False):
        if not (normalized):
            self.inputArray = np.insert(inputArray, 0, np.ones(len(inputArray)), axis = 0)
        else:
            self.inputArray = inputArray
        self.outputArray = outputArray
        self.indicators = indicatorArray
        self.p = len(self.inputArray[0] - 1)
        self.N = len(self.inputArray)
        self.K = len(self.outputArray[0])

    def standardizePredictor(self):
        X = np.transpose(np.transpose(self.inputArray)[1:])
        meanInput = np.mean(X ,axis=0)
        self.XMean = meanInput
        stdInput = np.std(X, axis=0)
        self.XStd = stdInput
        standardizedInput = (X - meanInput) / stdInput
        self.inputArray= np.insert(standardizedInput, 0, 1, axis=1)
    
    def bestFitRSS(self):
        xTy = np.dot(np.transpose(self.inputArray), self.outputArray)
        matrix = np.dot(np.transpose(self.inputArray), self.inputArray)
        realMatrix = linalg.inv(matrix)
        self.bestFit = np.dot(realMatrix, xTy)
        return self.bestFit


    def determineClassRegg(self, x):
        if not(hasattr(self, 'bestfit')):
            self.bestFitRSS()
        target = np.dot(x, self.bestFitRSS)
        indicator = np.argmax(target)
        return self.indicators[indicator]

    def setDiscriminants(self, probabilities=None, means=None, covariance=None):
        if probabilities=None
            probabilities = np.zeros(self.K)
            for ii in range(self.K):
                probabilities[ii] = np.sum(self.outputArray[:, ii])/self.N
        if means=None
            means = np.zeros(self.K)
            for ii in range(self.N):
                indication = np.where(self.outputArray[ii] == 1)[0] #there should always be exactly one 1
                means[indication] += self.inputArray[indication]/(probabilities[indication]*self.N)
        if covariance=None
            covariance = 0
            for kk in range(self.K):
                indications = np.where(self.outputArray[:, kk])
                for ii in indications:
                    covariance += (np.dot(self.inputArray[ii] - means[kk], self.inputArray[ii] - means[kk])/(self.N - self.K)
        self.probabilities = probabilities
        self.means = means
        self.covariance = covariance

    def linearDiscrinantAnalysisSolve(self, x):
        check = ['probabilities', 'means', 'covariance']
        needToApproximate = []
        for var in check:
            if not(hasattr(self, var)):
                needToApproximate.append(getattr(self, var))
            else:
                needToApproximate.append(None)
        self.setDiscriminants(check[0], check[1], check[2])
        deltas = np.zeros(self.K)
        for ii in range(self.K):
            deltas[ii] = np.dot(np.transpose(x), 1/covariance*self.means[ii]) - 1/2*np.dot(np.transpose(self.means[ii]), 1/covariance*self.means[ii]) + math.log(probabilities[ii])
        indicator = np.argmax(target)
        return self.indicators[indicator]

    def quadraticDiscriminatnAnalysisSolve(self, x):
        check = ['probabilities', 'means', 'covariance']
        needToApproximate = []
        for var in check:
            if not(hasattr(self, var)):
                needToApproximate.append(getattr(self, var))
            else:
                needToApproximate.append(None)
        self.setDiscriminants(check[0], check[1], check[2])
        deltas = np.zeros(self.K)
        for ii in range(self.K):
            -1/2*math.log(math.abs())
        
        
        

            


