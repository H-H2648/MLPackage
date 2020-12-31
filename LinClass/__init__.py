import numpy as np
import math
from scipy import linalg
from scipy.special import expit


class LinClass:
    #inputArray is of the form: 
    # assumes input is of the form [[x11, x12, x13, ..., x1p], [x21, x22, ..., x2p], [xN1, xN2, ..., xNp]] (numpy array)
    # assumes output is of the form [y1 | y2| y3|, ... , |yK] (numpy array)
    #where yij = 1 iff row j corresponds to the classification j
    #indicatorArray is of the form [G1, G2, ..., GK] where Gi corresponds to classification, i
    # propered implies that someone has already turned the input to be of the form [[1, x11, x12, x13, ..., x1p], [1, x21, x22, ..., x2p], [1, xN1, xN2, ..., xNp]]
    # normalized implies that someone has already turned the input to be of the form [[1, x11, x12, x13, ..., x1p], [1, x21, x22, ..., x2p], [1, xN1, xN2, ..., xNp]]
    def __init__(self, inputArray, outputArray, indicatorArray, normalized=False):
        self.inputArray = inputArray
        self.outputArray = outputArray
        self.indicators = indicatorArray
        self.p = len(self.inputArray[0])
        self.N = len(self.inputArray)
        self.K = len(self.outputArray[0])


#assumes bestFit has already been obtained from LinRegg
    def determineClassRegg(self, x, bestFit):
        target = np.dot(x, bestFit)
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
            means = np.zeros((self.K, self.p))
            for ii in range(self.N):
                indication = np.nonzero(self.outputArray[ii])[0][0] #there should always be exactly one 1
                means[indication] += self.inputArray[indication]/(probabilities[indication]*self.N)
            self.means = means
        else:
            self.means = val
    
    def setLinearCovariance(self, val=None):
        if val == None:
            if not(hasattr(self, 'means')):
                self.setMeans()
            means = self.means
            covariance = np.zeros((self.p, self.p))
            for kk in range(self.K):
                indications = np.nonzero(self.outputArray[:, kk])[0]
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
            covariances = np.zeros((self.K, self.p, self.p))
            for kk in range(self.K):
                indications = np.nonzero(self.outputArray[:, kk])[0]
                indicationsLength = len(indications)
                for ii in indications:
                    covariances[kk] += (np.outer(self.inputArray[ii] - means[kk], self.inputArray[ii] - means[kk])/(indicationsLength - 1))
            self.quadraticCovariance = covariances
        else:
            if len(val) != self.K:
                raise Exception('your covariances must match the classifiers: number of classifications {0}, covariance length {1}'.format(self.K, len(val))) 



    def LDASolve(self, x):
        if not (hasattr(self, 'probabilities')):
            self.setProbabilities()
        if not(hasattr(self, 'means')):
            self.setMeans
        if not(hasattr(self, 'linearCovariance')):
            self.setLinearCovariance()
        #UNOPTIMIZED SOLUTION
        deltas = np.zeros(self.K)
        for ii in range(self.K):
            deltas[ii] = np.dot(x, np.dot(linalg.inv(self.linearCovariance), self.means[ii])) - 1/2*np.dot(np.transpose(self.means[ii]), np.dot(linalg.inv(self.linearCovariance), self.means[ii])) + math.log(self.probabilities[ii])
        indicator = np.argmax(deltas)
        return self.indicators[indicator]

    def sphereInput(self):
        if not (hasattr(self, 'probabilities')):
            self.setProbabilities()
        if not(hasattr(self, 'means')):
            self.setMeans
        if not(hasattr(self, 'linearCovariance')):
            self.setLinearCovariance()
        U, d, Ut = np.linalg.svd(self.linearCovariance, hermitian=True)
        invSqrtD = np.diag(np.sqrt(1/d))
        X = np.transpose(np.dot(np.dot(invSqrtD, Ut), np.transpose(self.inputArray)))
        means = np.zeros((self.K, self.p))
        for ii in range(self.N):
            indication = np.nonzero(self.outputArray[ii])[0] #there should always be exactly one 1
            means[indication] += X[indication]/(self.probabilities[indication]*self.N)
        self.spheredMean = means
    
    def optimizedLDASolve(self, x):
        if not(hasattr(self, 'spheredMean')):
            self.sphereInput()
        means= self.spheredMean
        distanceFromMean = np.zeros(self.K)
        for ii in range(self.K):
            distanceFromMean[ii] = -1/2 * linalg.norm(x - means[ii])
        findMax = distanceFromMean + np.log(self.probabilities)
        indicator = np.argmax(findMax)
        return self.indicators[indicator]


    def QDASolve(self, x):
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

#don't really know if this is how it works
    def reducedRankLinearDiscriminantSolve(self, x):
        if not(hasattr(self, 'means')):
            self.setMeans()
        M = self.means
        if not(hasattr(self, 'linearCovariance')):
            self.setLinearCovariance()
        W = self.linearCovariance
        U, d, Ut = linalg.svd(self.linearCovariance, hermitian=True)
        invSqrtD = np.diag(np.sqrt(1/d))
        invSqrtW = np.dot(U, np.dot(invSqrtD, Ut))
        M = np.dot(M, invSqrtW)
        B = np.cov(M)
        V, D, Vt = linalg.svd(B, hermitian=True)
        Vectors = np.zeros((self.K, self.p))
        for ii in range(self.K):
            Vectors[ii] = np.dot(invSqrtW, Vt[:,ii])
        deltas = np.zeros(self.K)
        for ii in range(self.K):
            deltas[ii] = np.dot(Vectors[ii], x)
        indicator = np.argmax(deltas[ii])
        return self.indicators[indicator]


#assumes for output that [1 0 0 0 ... 0 ] represents negative and everything else represents positive
#y = 1 means 'True', y = 0 means 'False
    def binaryLogisticRegg(self, outputIndex=0):
        if not hasattr(self, 'XMean'):
            meanInput = np.mean(self.inputArray ,axis=0)
            self.XMean = meanInput
        if not hasattr(self, 'stdInput'):
            stdInput = np.std(self.inputArray, axis=0)
            self.XStd = stdInput
        if not hasattr(self, 'standardizedInput'):
            standardizedInput = (self.inputArray - meanInput) / stdInput
            self.standardizedInput= np.insert(standardizedInput, 0, 1, axis=1)
        y = np.zeros(self.N)
        for ii in range(len(self.outputArray)):
            if self.outputArray[ii][outputIndex] == 1:
                y[ii] = 1
            else:
                y[ii] = 0
        betaOrig = np.zeros(self.p + 1)
        currentProbabilities = 1 - expit(-np.dot(self.standardizedInput, betaOrig))
        otherProbabilities = 1 - currentProbabilities
        binomialResults = np.multiply(currentProbabilities, otherProbabilities)
        derivativeB = np.dot(np.transpose(self.standardizedInput), (y - currentProbabilities))
        XTW = np.copy(np.transpose(self.standardizedInput))
        for ii in range(len(binomialResults)):
            XTW[:,ii] = binomialResults[ii]*XTW[:,ii]
        doubleDerivativeB = -np.dot(XTW, self.standardizedInput)
        newBeta = betaOrig - np.dot(linalg.inv(doubleDerivativeB), derivativeB)
        step = 0
        distance = linalg.norm(newBeta - betaOrig)
        while distance > 0.0000005:
            betaOrig = newBeta
            currentProbabilities = 1 - expit(-np.dot(self.standardizedInput, betaOrig))
            otherProbabilities = 1 - currentProbabilities
            binomialResults = np.multiply(currentProbabilities, otherProbabilities)
            W = np.diag(binomialResults)
            derivativeB = np.dot(np.transpose(self.standardizedInput), (y - currentProbabilities))
            doubleDerivativeB = -np.dot(np.dot(np.transpose(self.standardizedInput), W), self.standardizedInput)
            newBeta = betaOrig - np.dot(linalg.inv(doubleDerivativeB), derivativeB)*(1/2**step)
            newDistance = linalg.norm(newBeta - betaOrig)
            if newDistance/distance > 0.9999:
                step +=1
            distance = newDistance
        return newBeta
    
    #introduce coordinate descent(?)
    def binaryLogisticReggLasso(self, complexity, outputIndex=0):
        if not hasattr(self, 'XMean'):
            meanInput = np.mean(self.inputArray ,axis=0)
            self.XMean = meanInput
        if not hasattr(self, 'stdInput'):
            stdInput = np.std(self.inputArray, axis=0)
            self.XStd = stdInput
        if not hasattr(self, 'standardizedInput'):
            standardizedInput = (self.inputArray - meanInput) / stdInput
            self.standardizedInput= np.insert(standardizedInput, 0, 1, axis=1)
        y = np.zeros(self.N)
        for ii in range(len(self.outputArray)):
            if self.outputArray[ii][outputIndex] == 1:
                y[ii] = 0
            else:
                y[ii] = 1
        betaOrig = np.zeros(self.p + 1)
        currentProbabilities = 1 - expit(-np.dot(self.standardizedInput, betaOrig))
        otherProbabilities = 1 - currentProbabilities
        binomialResults = np.multiply(currentProbabilities, otherProbabilities)
        W = np.diag(binomialResults)
        derivativeB = np.dot(np.transpose(self.standardizedInput), (y - currentProbabilities)) - np.sign(betaOrig)*complexity
        doubleDerivativeB = -np.dot(np.dot(np.transpose(self.standardizedInput), W), self.standardizedInput)
        newBeta = betaOrig - np.dot(linalg.inv(doubleDerivativeB), derivativeB)
        step = 0
        distance = linalg.norm(newBeta - betaOrig)
        while distance > 0.0000005:
            betaOrig = newBeta
            currentProbabilities = 1 - expit(-np.dot(self.standardizedInput, betaOrig))
            otherProbabilities = 1 - currentProbabilities
            binomialResults = np.multiply(currentProbabilities, otherProbabilities)
            W = np.diag(binomialResults)
            derivativeB = np.dot(np.transpose(self.standardizedInput), (y - currentProbabilities)) - np.sign(betaOrig)*complexity
            doubleDerivativeB = -np.dot(np.dot(np.transpose(self.standardizedInput), W), self.standardizedInput)
            newBeta = betaOrig - np.dot(linalg.inv(doubleDerivativeB), derivativeB)*(1/2**step)
            newDistance = linalg.norm(newBeta - betaOrig)
            if newDistance/distance > 0.9999:
                step +=1
            distance = newDistance
        return newBeta

    def logisticSolve(self, slope, x):
        probabilities = np.zeros(2)
        probabilities[0] = 1 - expit(-np.dot(x, slope))
        probabilities[1] = 1 - probabilities[0]
        indicator = np.argmax(probabilities)
        return self.indicators[indicator]


    

    def standardizeTest(self, testX):
        standardizedTestX = (testX - self.XMean)/self.XStd
        return np.insert(standardizedTestX, 0, 1, axis=1)

    #it doesn't really converge for some reason(?)
    def separateHyperPlane(self, outputIndex = 0, learningRate=1):
        if not hasattr(self, 'XMean'):
            meanInput = np.mean(self.inputArray ,axis=0)
            self.XMean = meanInput
        if not hasattr(self, 'stdInput'):
            stdInput = np.std(self.inputArray, axis=0)
            self.XStd = stdInput
        if not hasattr(self, 'standardizedInput'):
            standardizedInput = (self.inputArray - meanInput) / stdInput
            self.standardizedInput= np.insert(standardizedInput, 0, 1, axis=1)
        currentBeta = np.random.normal(size=self.p + 1)
        currentObs = np.dot(self.standardizedInput, currentBeta)
        plusMinusY = np.zeros(self.N)
        for ii in range(self.N):
            if self.outputArray[ii][outputIndex] == 1:
                plusMinusY[ii] = 1
            else:
                plusMinusY[ii] = -1
        findWrong = np.not_equal(np.sign(plusMinusY), np.sign(currentObs)).nonzero()[0]
        print(len(findWrong))
        while len(findWrong) > 0:
            for ii in findWrong:
                currentBeta += np.insert(learningRate*(plusMinusY[ii]*self.standardizedInput[ii][1:]), 0, plusMinusY[ii])
            currentObs = np.dot(self.standardizedInput, currentBeta)
            findWrong = np.not_equal(np.sign(plusMinusY), np.sign(currentObs)).nonzero()[0]
            print(len(findWrong))


    


        
        

            


