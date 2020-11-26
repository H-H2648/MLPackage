import numpy as np
import math
from scipy import linalg
from scipy.stats import norm, t, f, pearsonr
from scipy.sparse.linalg import lsqr


class LinRegg:
    # assumes input is of the form [[x11, x12, x13, ..., x1p], [x21, x22, ..., x2p], [xN1, xN2, ..., xNp]] (numpy array)
    # assumes output is of the form [y1, y2, y2, ... , yN] (numpy array)
    # propered implies that someone has already turned the input to be of the form [[1, x11, x12, x13, ..., x1p], [1, x21, x22, ..., x2p], [1, xN1, xN2, ..., xNp]]

    # THINGS TO DO LATER:
    # inputArray and outputArray should have the same size
    # check if inputArray is linearly independent [remove 'duplicate']
    def __init__(self, inputArray, outputArray, propered=False):
        if not (propered):
            inputLst = inputArray.tolist()
            properLst = []
            for row in inputLst:
                properLst.append([1] + row)
            self.inputArray = np.array(properLst)
        else:
            self.inputArray = inputArray
        self.outputArray = outputArray
        print(self.outputArray)
        self.p = len(self.inputArray[0]) - 1
        self.N = len(self.inputArray)

#removes the bias term 1 first before standardizing

    def standardizePredictor(self):
        X = np.transpose(np.transpose(self.inputArray)[1:])
        meanInput = np.mean(X ,axis=0)
        stdInput = np.std(X, axis=0)
        standardizedInput = (X - meanInput) / stdInput
        self.standardizedInputArray= np.insert(standardizedInput, 0, 1, axis=1)

    def RSSSolve(self):
        xTy = np.dot(np.transpose(self.inputArray), self.outputArray)
        matrix = np.dot(np.transpose(self.inputArray), self.inputArray)
        realMatrix = linalg.inv(matrix)
        self.bestFit = np.dot(realMatrix, xTy)
        return self.bestFit

    def residual(self):
        if not (hasattr(self, 'bestFit')):
            self.RSSSolve()
        testOutput = np.dot(self.inputArray, self.bestFit)
        return (self.outputArray - testOutput)

    # assumes input is simply [x1, x2, ..., xp]
    # propered implies that someone sends the input [1, x1, x2, ..., xp]
    def predict(self, inputVector, propered=False):
        bestFitLine = self.RSSSolve()
        if not (propered):
            inputLst = inputVector.tolist()
            properLst = [1] + inputLst
            properVector = np.array(properLst)
        else:
            properVector = inputVector
        if not (hasattr(self, 'bestFit')):
            return np.dot(properVector, self.bestFit)
        else:
            self.RSSSolve()
            return np.dot(properVector, self.bestFit)

    def RSS(self):
        if not (hasattr(self, 'bestFit')):
            self.RSSSolve()
        self.RSSVal = np.dot(self.outputArray - np.dot(self.inputArray, self.bestFit), self.outputArray - np.dot(self.inputArray, self.bestFit))
        return (self.RSSVal)

    def setVariance(self, variance):
        self.variance = variance

    def approximateVariance(self):
        self.variance = 1 / (self.N - self.p - 1) * (np.dot(self.residual(), self.residual()))
        return self.variance

    # Assumes N (the sample size) is very large
    # checkMatters calculates whether Bi = 0 (and thus the variable doesn't matter) or not
    # Note False does not mean that it doesn't matter. It is simply indeterminant
    def zScore(self, index, variance=None):
        if not (hasattr(self, 'bestFit')):
            self.RSSSolve()
        if variance == None:
            if not (hasattr(self, 'variance')):
                self.approximateVariance()
            variance = self.variance
        normalizedStandardDeviationMatrix = linalg.inv(np.dot(np.transpose(self.inputArray), self.inputArray))
        zScore = self.bestFit[index] / (math.sqrt(variance * normalizedStandardDeviationMatrix[index][index]))
        return (zScore)

    def checkMatters(self, index, variance=None, pValue=0.05, tTest=False):
        zScore = self.zScore(self, index, variance)
        if tTest:
            rv = t(df=len(self.N - self.p - 1))
        else:
            rv = norm()
        p = rv.sf(zScore)
        if p > 1 - (pValue) / 2 or p < (pValue) / 2:
            return True
        else:
            return False

    # assumes 0 is not in excludeLst
    def FTest(self, excludeLst, pValue=0.05):
        subsetInputArray = np.delete(self.inputArray, excludeLst, axis=1)
        subsetLinTest = LinRegg(subsetInputArray, self.outputArray, propered=True)
        subsetLinTest.RSS()
        if not (hasattr(self, 'RSSVal')):
            self.RSS()
        fNumerator = (subsetLinTest.RSSVal - self.RSSVal) / (self.p - subsetLinTest.p)
        fDenominator = self.RSSVal / (self.N - self.p - 1)
        self.fScore = fNumerator / fDenominator
        rv = f(dfn=self.p - subsetLinTest.p, dfd=self.N - self.p - 1)
        p = rv.sf(self.fScore)
        if p > 1 - (pValue) / 2 or p < (pValue) / 2:
            return True
        else:
            return False

    def orthogonalizationAlgorithm(self):
        x = np.transpose(self.inputArray)
        bestFit = np.zeros(self.p+1)
        z = np.ones(self.N)
        zArray = np.zeros(shape = (self.p + 1, self.N))
        x = np.transpose(self.inputArray)
        zArray[0] = z
        bestFit[0] = np.dot(zArray[0], self.outputArray)/np.dot(zArray[0], zArray[0])
        for ii in range(1, self.p + 1):
            z = x[ii]
            for jj in range(0, ii):
                z -= (np.dot(zArray[jj], x[ii])/np.dot(zArray[jj], zArray[jj])) * z[jj]
            zArray[ii] = z
            bestFit[ii] = (np.dot(zArray[ii], self.outputArray)/np.dot(zArray[jj], zArray[jj]))
        self.orthogonalBestFit = bestFit
        self.orthogonalResiduals = zArray
        return bestFit

    def orthogonalVarianceEstimate(self, variance=None):
        if not(hasattr(self, 'orthogonalResiduals')):
            self.orthogonalizationAlgorithm()
        if not(hasattr(self, 'variance')):
            self.approximateVariance()
        self.orthogonalVarianceEstimate = self.variance/(np.dot(self.orthogonalResidiauls[-1], self.orthogonalResidiauls[-1]))
        return self.orthogonalVarianceEstimate

    #for now only accounts for one-dimensional output
    #complexity is the complexity parameter (lambda)
    #highlights the variables with higher variance
    def RSSRidgeSolve(self, complexity):
        if self.outputArray.ndim > 1:
            beta0 = np.mean(self.outputArray, axis=1)
        else:
            beta0 = np.mean(self.outputArray)
        X = np.transpose(np.transpose(self.inputArray)[1:])
        xTy = np.dot(np.transpose(X), self.outputArray)
        matrix = np.dot(np.transpose(X), X) + complexity*np.identity(self.p)
        realMatrix = linalg.inv(matrix)
        Beta = np.dot(realMatrix, xTy)
        self.ridgeBestFit = np.insert(Beta, 0, beta0, axis = 0)
        return self.ridgeBestFit

    #assumes single output learning for now
    #epsilon is so far just random
    def LARAlgorithm(self, alpha=0.1):
        #helper function
        #assumes elem is not in the lst
        def findIndexInSortedLst(elem, lst):
            val = 0
            while lst != []:
                if elem < lst[0]:
                    return val
                elif elem > lst[-1]:
                    return val + len(lst)-1
                else:
                    half = int(len(lst))/2
                    if elem < lst[half]:
                        lst = lst[:half]
                    elif elem > lst[half]:
                        lst = lst[half+1:]
                        val += half
        if not (hasattr(self, 'standardizedInputArray')):
            self.standardizePredictor()
        X = self.standardizedInputArray
        activeXT = np.array([[1]*self.N])
        y = self.outputArray
        beta0 = np.mean(y)
        bestFit = np.zeros(1)
        bestFit[0] = beta0
        mostCorrelatedIndices = [0]
        for ii in range(min(self.p, self.N-1)):
            residual = y -np.dot(np.transpose(activeXT), bestFit)
            while True:
                curMaxCor = -1
                maxCorrelatedIndex = None
                XTranspose = X.transpose()
                for jj in range(1, len(XTranspose)):
                    correlation = abs(pearsonr(XTranspose[jj], residual)[0])
                    if correlation > curMaxCor:
                        maxCorrelatedIndex = jj
                        curMaxCor = correlation
                if not maxCorrelatedIndex in mostCorrelatedIndices:
                    place = findIndexInSortedLst(maxCorrelatedIndex, mostCorrelatedIndices)
                    activeXTLst = activeXT.tolist()
                    activeXTLst.insert(place+1, X[:, maxCorrelatedIndex].tolist())
                    activeXT = np.array(activeXTLst)
                    bestFit = np.insert(bestFit, place, 0)
                    mostCorrelatedIndices.insert(place, maxCorrelatedIndex)
                    break
                else:
                    delta = np.dot((np.dot(linalg.inv(np.dot(activeXT, np.transpose(activeXT))), activeXT)), residual)
                    bestFit += alpha*delta


                residual = y - np.dot(np.transpose(activeXT), bestFit)
        self.LARBestFit = bestFit
        return self.LARBestFit


    #same as above but adds the condition that when non-zero coefficient hits zero, drop its variable from the active set of variable
    def LARLassoAlgorithm(self, alpha=0.1):
        #helper function
        #assumes elem is not in the lst
        def findIndexInSortedLst(elem, lst):
            val = 0
            while lst != []:
                if elem < lst[0]:
                    return val
                elif elem > lst[-1]:
                    return val + len(lst)-1
                else:
                    half = int(len(lst))/2
                    if elem < lst[half]:
                        lst = lst[:half]
                    elif elem > lst[half]:
                        lst = lst[half+1:]
                        val += half
        if not (hasattr(self, 'standardizedInputArray')):
            self.standardizePredictor()
        X = self.standardizedInputArray
        activeXT = np.array([[1]*self.N])
        y = self.outputArray
        beta0 = np.mean(y)
        bestFit = np.zeros(1)
        bestFit[0] = beta0
        mostCorrelatedIndices = [0]
        for ii in range(min(self.p, self.N-1)):
            residual = y -np.dot(np.transpose(activeXT), bestFit)
            while True:
                curMaxCor = -1
                maxCorrelatedIndex = None
                XTranspose = X.transpose()
                for jj in range(1, len(XTranspose)):
                    correlation = abs(pearsonr(XTranspose[jj], residual)[0])
                    if correlation > curMaxCor:
                        maxCorrelatedIndex = jj
                        curMaxCor = correlation
                if not maxCorrelatedIndex in mostCorrelatedIndices:
                    place = findIndexInSortedLst(maxCorrelatedIndex, mostCorrelatedIndices)
                    activeXTLst = activeXT.tolist()
                    activeXTLst.insert(place+1, X[:, maxCorrelatedIndex].tolist())
                    activeXT = np.array(activeXTLst)
                    bestFit = np.insert(bestFit, place, 0)
                    mostCorrelatedIndices.insert(place, maxCorrelatedIndex)
                    break
                else:
                    delta = np.dot((np.dot(linalg.inv(np.dot(activeXT, np.transpose(activeXT))), activeXT)), residual)
                    bestFit += alpha*delta
                    jj = 0
                    while jj < len(bestFit):
                        if bestFit[jj] == 0:
                            print(bestFit)
                            bestFit = np.delete(bestFit, jj)
                            print(bestFit)
                            activeXT = np.delete(activeXT, jj, 0)
                            print('removed: {0}'.format(mostCorrelatedIndices[jj]))
                        jj += 1

                residual = y - np.dot(np.transpose(activeXT), bestFit)
        self.lassoBestFit = bestFit
        return self.lassoBestFit

    #things to do:
    #what is the svd algorithm? Can it be more efficient?
    #gives solution to standardized input
    def principalComponentRegression(self):
        if not(hasattr(self, 'standardizedInputArray')):
            self.standardizePredictor()
        X = self.standardizedInputArray
        y = self.outputArray
        #only interested in a columns of eigenvectors, vh
        u, s, vh = linalg.svd(X)
        bestFit = np.zeros(self.p+1)
        for v in vh:
            z = np.dot(X, v)
            theta = np.dot(z, y)/np.dot(z, z)
            bestFit +=  theta*v
        self.principalComponentBestFit = bestFit
        return (self.principalComponentBestFit)

    def partialLeastSquares(self, M= None):
        if M == None:
            M = self.p
        if not(hasattr(self, 'standardizedInputArray')):
            self.standardizePredictor()
        X = self.standardizedInputArray
        #tranposed X without the 1s
        superX = np.zeros((self.p, self.N))
        y = self.outputArray
        Y = np.zeros((M + 1, self.N))
        Y[0] = np.ones(self.N)*np.mean(self.outputArray)
        for ii in range(self.p):
            superX[ii] = X[:,ii+1]
        phisArray = np.zeros(self.p)
        for ii in range(M):
            for jj in range(self.p):
                phisArray[jj] = np.dot(superX[jj], self.outputArray)
            z = np.zeros(self.N)
            for jj in range(self.p):
                z += phisArray[jj]*superX[jj]
            theta = np.dot(z, self.outputArray)/np.dot(z, z)
            Y[ii+1] = y[ii] + theta*z
            for jj in range(self.p):
                superX[jj] -= np.dot(z, superX[jj])/np.dot(z, z)*z

        self.partialLeastSquaresBestFit = lsqr(X, Y[M])
        return(self.partialLeastSquaresBestFit)

    #OVERALL SUMMARY:
    #Ridge is generally the most preferable for minimizing prediction errors
    #PLS, PCR and Ridge are fairly similar
    #Lasso is somewhere between Ridge and best subsets (enjoys favourable properties of both)



