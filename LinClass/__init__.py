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
        
    def setProbabilities(self, val=None):
        if val is None:
            probabilities = np.zeros(self.K)
            for ii in range(self.K):
                probabilities[ii] = np.sum(self.outputArray[:, ii])/self.N
            self.probabilities = probabilities
        else:
            if len(val) != self.K:
                raise Exception('your probabilities must match the classifiers: number of classifications {0}, number of probabilities {1}'.format(self.K, len(val))) 
            self.probabilities = val

    def setMeans(self, val = None):
        if val == None:
            if not(hasattr(self, 'probabilities')):
                self.setProbabilities()
                probabilities = self.probabilities
            means = np.zeros(self.K)
            for ii in range(self.N):
                indication = np.nonzero(self.outputArray[ii])[0] #there should always be exactly one 1
                means[indication] += self.inputArray[indication]/(probabilities[indication]*self.N)
            self.means = means
        else:
            self.means = val
    
    def setLinearCovariance(self, val=None):
        if val == None:
            if not(hasattr(self, 'means')):
                self.setMeans()
                means = self.means
            covariance = np.zeros((self.N, self.N))
            for kk in range(self.K):
                indications = np.nonzero(self.outputArray[:, kk])
                for ii in indications:
                    covariance += (np.outer(self.inputArray[ii] - means[kk], self.inputArray[ii] - means[kk])/(self.N - self.K))
            self.linearCovariance = covariance
        else:
            self.linearCovariance = val
    
    def setQuadraticCovariance(self, val = None):
        if val == None:
            if not(hasattr(self, 'means')):
                self.setMeans()
                means = self.means
            covariances = np.zeros((self.K, self.N, self.N))
            for kk in range(self.K):
                indications = np.nonzero(self.outputArray[:, kk])
                indicationsLength = len(indications)
                for ii in indications:
                    covariances[kk] += (np.outer(self.inputArray[ii] - means[kk], self.inputArray[ii] - means[kk])/(indicationsLength - 1))
            self.quadraticCovariance = covariances
        else:
            if len(val) != self.K:
                raise Exception('your covariances must match the classifiers: number of classifications {0}, covariance length {1}'.format(self.K, len(val))) 



    def linearDiscrinantAnalysisSolve(self, x):
        if not (hasattr(self, 'probabilities')):
            self.setProbabilities()
        if not(hasattr(self, 'means')):
            self.setMeans
        if not(hasattr(self, 'linearCovariance')):
            self.setLinearCovariance()
        #UNOPTIMIZED SOLUTION
        #deltas = np.zeros(self.K)
        #for ii in range(self.K):
        #    deltas[ii] = np.dot(x, np.dot(linalg.inv(self.linearCovariance), self.means[ii])) - 1/2*np.dot(np.transpose(self.means[ii]), linalg.inv(self.linearCovariance)*self.means[ii]) + math.log(self.probabilities[ii])
        #indicator = np.argmax(deltas)
        #OPTIMIZED SOLUTION
        U, d, Ut = np.linalg.svd(self.linearCovariance, hermitian=True)
        invSqrtD = np.diag(np.sqrt(1/d))
        X = np.dot(invSqrtD, np.dot(Ut, self.inputArray))
        means = np.zeros(self.K)
        for ii in range(self.N):
            indication = np.nonzero(self.outputArray[ii])[0] #there should always be exactly one 1
            means[indication] += X[indication]/(self.probabilities[indication]*self.N)
        distanceFromMean = np.zeros(self.K)
        for ii in range(self.K):
            distanceFromMean[ii] = -1/2 * np.norm(x - self.means[ii])
        findMin = distanceFromMean + np.log(self.probabilities)
        indicator = np.argmax(findMin)
        return self.indicators[indicator]


    def quadraticDiscriminantAnalysisSolve(self, x):
        if not (hasattr(self, 'probabilities')):
            self.setProbabilities()
        if not(hasattr(self, 'means')):
            self.setMeans
        if not(hasattr(self, 'quadraticCovariance')):
            self.setQuadraticCovariance()
        deltas = np.zeros(self.K)
        for ii in range(self.K):
           deltas[ii] = -1/2*math.log(linalg.norm(self.quadraticCovariance[ii])) - 1/2*np.dot(x - self.means[ii], np.dot(linalg.inv(self.quadraticCovariance[ii]), x - self.means[ii])) + math.log(self.probabilities[ii])
        indicator = np.argmax(deltas)
        return self.indicators[indicator]
        

#maybe optimize this(?)
#memoization? or would that cause memory leak when trying to find the best alpha?
#have memoization variable?
    def regularlizeDiscriminantAnalysisSolve(self, alpha, x):
        if not(hasattr(self, 'linearCovariance')):
            self.setLinearCovariance()
        if not(hasattr(self, 'quadraticCovariance')):
            self.setQuadraticCovariance()
        covariance = alpha*self.quadraticCovariance + (1-alpha)*np.dot(self.linearCovariance, np.ones(self.K))
        deltas = np.zeros(self.K)
        for ii in range(self.K):
           deltas[ii] = -1/2*math.log(linalg.norm(covariance[ii])) - 1/2*np.dot(x - self.means[ii], np.dot(linalg.inv(covariance[ii]), x - self.means[ii])) + math.log(self.probabilities[ii])
        indicator = np.argmax(deltas[ii])
        return self.indicators[indicator]
    


        
        

            


