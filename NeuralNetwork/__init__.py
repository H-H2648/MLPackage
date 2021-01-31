import numpy as np
import math
from scipy import linalg
from scipy.special import expit
from scipy.special import softmax

#it works! :)

#for now single hidden layer neural-network

#very simple neural network for with activation function = sigmoid, output = identity, measure of fit = sum-of-squared error, using ridge weight decay
#derivative of expit = expit*(1-expit)
class SimpleRegressionNeuralNetwork:
    
    def __init__(self, inputArray, outputArray, hiddenNeurons, weightDecayRate=0, normalized=False):
        self.inputArray = inputArray
        self.outputArray = outputArray
        if not (normalized):
            self.inputArray = np.insert(inputArray, 0, np.ones(len(inputArray)), axis = 1)
        else:
            self.inputArray = inputArray
        self.p = len(self.inputArray[0]) - 1
        self.M = hiddenNeurons
        self.N = len(self.outputArray)
        if self.outputArray.ndim > 1:
            self.K = len(self.outputArray[0])
        else:
            self.K = 1
        self.weightDecay = weightDecayRate

    def prepareTrain(self):
        X = np.transpose(np.transpose(self.inputArray)[1:])
        meanInput = np.mean(X ,axis=0)
        self.XMean = meanInput
        stdInput = np.std(X, axis=0)
        self.XStd = stdInput
        standardizedInput = (X - meanInput) / stdInput
        self.inputArray= np.insert(standardizedInput, 0, 1, axis=1)
        self.alphas = np.random.uniform(low=-0.7, high = 0.7, size=((self.M, self.p + 1)))
        if self.outputArray.ndim > 1:
            self.betas = np.random.uniform(low=-0.7, high = 0.7, size = ((self.K, self.M + 1)))
        else:
            self.betas = np.random.uniform(low = -0.7, high = 0.7, size=self.M + 1)

    def feed(self, learningRate, batchSize=64):
        indices = np.random.choice(self.inputArray.shape[0], batchSize, replace=False)
        X = self.inputArray[indices]
        y = self.outputArray[indices]
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.transpose(np.dot(self.betas, np.transpose(Z)))
        predictedVal = T
        if self.outputArray.ndim > 1:
            dAs = np.zeros((self.M, self.p+1))
            dBs = np.zeros(self.K, self.M + 1)
            for mm in range(self.M):
                for ll in range(self.p + 1):
                    dRa = np.zeros(len(X))
                    derivG = 1
                    sigma = expit(np.dot(X, self.alphas[mm]))
                    derivSigma = np.multiply(sigma, 1- sigma)
                    for kk in range(self.K):
                        differenceK = y[:, kk] - predictedVal[:, kk]
                        dRa -= np.multiply(2*differenceK*derivG*self.betas[mm], np.multiply*derivSigma, X[:, ll])
                    self.alphas[mm][ll] -= learningRate*(np.average(dRa) +2*self.alphas[mm][ll]*self.weightDecay)
            for kk in range(self.K):
                differenceK = y[:, kk] - predictedVal[:, kk]
                for mm in range(self.M + 1):
                    dRB = -np.multiply(2*differenceK, Z[:,mm])
                    self.betas[kk][mm] -= learningRate*(np.average(dRB) + 2*self.betas[kk][mm]*self.weightDecay)
        else:
            dAs = np.zeros((self.M, self.p+1))
            dBs = np.zeros(self.M + 1)
            difference = y - predictedVal
            print(1/2 * np.dot(difference, difference)/len(X))
            for mm in range(self.M):
                for ll in range(self.p + 1):
                    derivG = 1
                    sigma = expit(np.dot(X, self.alphas[mm]))
                    derivSigma = np.multiply(sigma, 1- sigma)
                    dRa =  np.multiply(-2*difference*derivG*self.betas[mm], np.multiply(derivSigma, X[:,ll]))
                    dAs[mm][ll] = np.sum(dRa)
                    self.alphas[mm][ll] -= learningRate*(np.average(dRa) + 2*self.alphas[mm][ll]*self.weightDecay)
                    #self.alphas[mm][ll] -= learningRate*(np.average(dRa) + np.sign(self.alphas[mm][ll])*self.weightDecay)
                    #self.alphas[mm][ll] -= learningRate*(np.average(dRa) + 2*self.alphas[mm][ll]/((1 + (self.alphas[mm][ll])**2)**2)*self.weightDecay)
            for mm in range(self.M + 1):
                dRB = -np.multiply(2*difference, Z[:,mm])
                dBs[mm] = np.sum(dRB)
                self.betas[mm] -= learningRate*(np.average(dRB) + 2*self.betas[mm] * self.weightDecay)
                #self.betas[mm] -= learningRate*(np.average(dRB) + np.sign(self.betas[mm]) * self.weightDecay)
                #self.betas[mm] -= learningRate*(np.average(dRa) + 2*self.betas[mm]/((1 + (self.betas[mm])**2)**2)*self.weightDecay)
        #print('Alpha Derivatives:')
        #print(linalg.norm(dAs))
        #print('Beta Derivatives:')
        #print(linalg.norm(dBs))
        #print('Current Norms')
        #print(1/2 * linalg.norm(difference)**2/len(self.inputArray))
        return 1/2 * linalg.norm(difference)**2/len(X)


    def standardizeTest(self, testX):
        standardizedTestX = (testX - self.XMean)/self.XStd
        return np.insert(standardizedTestX, 0, 1, axis=1)

    #assumes x is stadardized
    def predict(self, X):
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.dot(self.betas, np.transpose(Z))
        predictedVal = T
        return predictedVal

class EntropyClassificationNeuralNetwork:
    
    def __init__(self, inputArray, outputArray, indicators, hiddenNeurons, weightDecayRate=0, normalized=False):
        self.inputArray = inputArray
        self.outputArray = outputArray
        self.indicators = indicators
        if not (normalized):
            self.inputArray = np.insert(inputArray, 0, np.ones(len(inputArray)), axis = 1)
        else:
            self.inputArray = inputArray
        self.p = len(self.inputArray[0]) - 1
        self.M = hiddenNeurons
        self.N = len(self.outputArray)
        self.K = len(self.outputArray[0])
        self.weightDecay = weightDecayRate

    def prepareTrain(self):
        X = np.transpose(np.transpose(self.inputArray)[1:])
        meanInput = np.mean(X ,axis=0)
        self.XMean = meanInput
        stdInput = np.std(X, axis=0)
        self.XStd = stdInput
        standardizedInput = (X - meanInput) / stdInput
        self.inputArray= np.insert(standardizedInput, 0, 1, axis=1)
        self.alphas = np.random.uniform(low=-0.7, high = 0.7, size=((self.M, self.p + 1)))
        if self.outputArray.ndim > 1:
            self.betas = np.random.uniform(low=-0.7, high = 0.7, size = ((self.K, self.M + 1)))
        else:
            self.betas = np.random.uniform(low = -0.7, high = 0.7, size=self.M + 1)

    def feed(self, learningRate, batchSize=64):
        #indices = np.random.choice(self.inputArray.shape[0], batchSize, replace=False)
        #X = self.inputArray[indices]
        #y = self.outputArray[indices]
        X = self.inputArray
        y = self.outputArray
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.dot(Z, np.transpose(self.betas))
        predictedVal = softmax(T, axis=1)
        print(y)
        print(predictedVal)
        costError = 0
        for kk in range(self.K):
            costError -= np.dot(y[:, kk], np.log(predictedVal[:, kk]))
        #costError = costError/len(X)
        dAs = np.zeros((self.M, self.p+1))
        dBs = np.zeros((self.K,self.M + 1))
        print('Current cost error: {0}'.format(costError))
        for mm in range(self.M):
            for ll in range(self.p + 1):
                dRa = np.zeros(len(X))
                sigma = expit(np.dot(X, self.alphas[mm]))
                derivSigma = np.multiply(sigma, 1- sigma)
                for kk in range(self.K):
                    ratioK = np.divide(y[:,kk], predictedVal[:,kk])
                    derivG = np.multiply(predictedVal[:,kk], 1 - predictedVal[:,kk])
                    dRa -= np.multiply(np.multiply(np.multiply(ratioK, derivG)*self.betas[kk][mm], derivSigma), X[:,ll])
                self.alphas[mm][ll] -= learningRate*(np.sum(dRa) +2*self.alphas[mm][ll]*self.weightDecay)
                dAs[mm][ll] = np.average(dRa)
        for kk in range(self.K):
            for mm in range(self.M + 1):
                ratioK = np.divide(y[:,kk], predictedVal[:,kk])
                derivG = np.multiply(predictedVal[:,kk], 1 - predictedVal[:,kk])
                dRB = -np.multiply(np.multiply(ratioK, derivG), Z[:,mm])
                self.betas[kk][mm] -= learningRate*(np.sum(dRB) + 2*self.betas[kk][mm]*self.weightDecay)
                dBs[kk][mm] = np.average(dRB)
        #print('Alpha Derivatives:')
        #print(linalg.norm(dAs))
        #print('Beta Derivatives:')
        #print(linalg.norm(dBs))


    def standardizeTest(self, testX):
        standardizedTestX = (testX - self.XMean)/self.XStd
        return np.insert(standardizedTestX, 0, 1, axis=1)

    #assumes x is stadardized
    def predict(self, X):
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.transpose(np.dot(self.betas, np.transpose(Z)))
        #predictedVal = softmax(T, axis=1)
        predictedVal = T
        results = np.zeros(len(X))
        bestArg = np.argmax(predictedVal, axis=1)
        for ii in range(len(results)):
            results[ii] = self.indicators[bestArg[ii]]
        return results

class SquaredClassificationNeuralNetwork:
    
    def __init__(self, inputArray, outputArray, indicators, hiddenNeurons, weightDecayRate=0, normalized=False):
        self.inputArray = inputArray
        self.outputArray = outputArray
        self.indicators = indicators
        if not (normalized):
            self.inputArray = np.insert(inputArray, 0, np.ones(len(inputArray)), axis = 1)
        else:
            self.inputArray = inputArray
        self.p = len(self.inputArray[0]) - 1
        self.M = hiddenNeurons
        self.N = len(self.outputArray)
        self.K = len(self.outputArray[0])
        self.weightDecay = weightDecayRate

    def prepareTrain(self):
        X = np.transpose(np.transpose(self.inputArray)[1:])
        meanInput = np.mean(X ,axis=0)
        self.XMean = meanInput
        stdInput = np.std(X, axis=0)
        self.XStd = stdInput
        standardizedInput = (X - meanInput) / stdInput
        self.inputArray= np.insert(standardizedInput, 0, 1, axis=1)
        self.alphas = np.random.uniform(low=-0.7, high = 0.7, size=((self.M, self.p + 1)))
        if self.outputArray.ndim > 1:
            self.betas = np.random.uniform(low=-0.7, high = 0.7, size = ((self.K, self.M + 1)))
        else:
            self.betas = np.random.uniform(low = -0.7, high = 0.7, size=self.M + 1)

    def feed(self, learningRate, batchSize=64):
        #indices = np.random.choice(self.inputArray.shape[0], batchSize, replace=False)
        #X = self.inputArray[indices]
        #y = self.outputArray[indices]
        X = self.inputArray
        y = self.outputArray
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.dot(Z, np.transpose(self.betas))
        #predictedVal = softmax(T, axis=1)
        predictedVal = T
        costError = 0
        for kk in range(self.K):
            costError += np.dot(y[:, kk] - predictedVal[:, kk], y[:, kk] - predictedVal[:, kk])
        costError = costError
        print('Current cost error: {0}'.format(costError))
        for mm in range(self.M):
            for ll in range(self.p + 1):
                dRa = np.zeros(len(X))
                sigma = expit(np.dot(X, self.alphas[mm]))
                derivSigma = np.multiply(sigma, 1- sigma)
                for kk in range(self.K):
                    differenceK = y[:,kk] - predictedVal[:,kk]
                    #derivG = np.multiply(predictedVal[:,kk], 1 - predictedVal[:,kk])
                    derivG = 1
                    dRa -= np.multiply(np.multiply(np.multiply(2*differenceK, derivG)*self.betas[kk][mm], derivSigma), X[:,ll])
                self.alphas[mm][ll] -= learningRate*(np.sum(dRa) +2*self.alphas[mm][ll]*self.weightDecay)
        for kk in range(self.K):
            for mm in range(self.M + 1):
                differenceK = y[:, kk] - predictedVal[:, kk]
                #derivG = np.multiply(predictedVal[:,kk], 1 - predictedVal[:,kk])
                derivG = 1
                dRB = -np.multiply(np.multiply(2*differenceK, derivG), Z[:,mm])
                self.betas[kk][mm] -= learningRate*(np.sum(dRB) + 2*self.betas[kk][mm]*self.weightDecay)



    def standardizeTest(self, testX):
        standardizedTestX = (testX - self.XMean)/self.XStd
        return np.insert(standardizedTestX, 0, 1, axis=1)

    #assumes x is stadardized
    def predict(self, X):
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.transpose(np.dot(self.betas, np.transpose(Z)))
        #predictedVal = softmax(T, axis=1)
        predictedVal = T
        results = np.zeros(len(X))
        bestArg = np.argmax(predictedVal, axis=1)
        for ii in range(len(results)):
            results[ii] = self.indicators[bestArg[ii]]
        return results

class TestClassificationNeuralNetwork:
    
    def __init__(self, inputArray, outputArray, indicators, hiddenNeurons, weightDecayRate=0, normalized=False):
        self.inputArray = inputArray
        self.outputArray = outputArray
        self.indicators = indicators
        if not (normalized):
            self.inputArray = np.insert(inputArray, 0, np.ones(len(inputArray)), axis = 1)
        else:
            self.inputArray = inputArray
        self.p = len(self.inputArray[0]) - 1
        self.M = hiddenNeurons
        self.N = len(self.outputArray)
        self.K = len(self.outputArray[0])
        self.weightDecay = weightDecayRate

    def prepareTrain(self):
        X = np.transpose(np.transpose(self.inputArray)[1:])
        meanInput = np.mean(X ,axis=0)
        self.XMean = meanInput
        stdInput = np.std(X, axis=0)
        self.XStd = stdInput
        standardizedInput = (X - meanInput) / stdInput
        self.inputArray= np.insert(standardizedInput, 0, 1, axis=1)
        self.alphas = np.random.uniform(low=-0.7, high = 0.7, size=((self.M, self.p + 1)))
        if self.outputArray.ndim > 1:
            self.betas = np.random.uniform(low=-0.7, high = 0.7, size = ((self.K, self.M + 1)))
        else:
            self.betas = np.random.uniform(low = -0.7, high = 0.7, size=self.M + 1)

    def feed(self, learningRate, batchSize=64):
        #indices = np.random.choice(self.inputArray.shape[0], batchSize, replace=False)
        #X = self.inputArray[indices]
        #y = self.outputArray[indices]
        X = self.inputArray
        y = self.outputArray
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.dot(Z, np.transpose(self.betas))
        predictedVal = expit(T)
        print(predictedVal)
        costError = 0
        for kk in range(self.K):
            costError += (np.dot(y[:, kk] - predictedVal[:, kk], y[:, kk] - predictedVal[:, kk])*2)/len(X)
        costError += self.weightDecay*(np.trace(np.dot(np.transpose(self.alphas), self.alphas)) + np.trace(np.dot(np.transpose(self.betas), self.betas)))
        print('Current cost error: {0}'.format(costError))
        for mm in range(self.M):
            for ll in range(self.p + 1):
                dRa = np.zeros(len(X))
                sigma = expit(np.dot(X, self.alphas[mm]))
                derivSigma = np.multiply(sigma, 1- sigma)
                for kk in range(self.K):
                    differenceK = y[:,kk] - predictedVal[:,kk]
                    derivG = np.multiply(predictedVal[:,kk], 1 - predictedVal[:,kk])
                    #derivG = 1
                    dRa -= np.multiply(np.multiply(np.multiply(differenceK, derivG)*self.betas[kk][mm], derivSigma), X[:,ll])
                self.alphas[mm][ll] -= learningRate*(np.sum(dRa) +2*self.alphas[mm][ll]*self.weightDecay)
        for kk in range(self.K):
            for mm in range(self.M + 1):
                differenceK = y[:, kk] - predictedVal[:, kk]
                derivG = np.multiply(predictedVal[:,kk], 1 - predictedVal[:,kk])
                #derivG = 1
                dRB = -np.multiply(np.multiply(differenceK, derivG), Z[:,mm])
                self.betas[kk][mm] -= learningRate*(np.sum(dRB) + 2*self.betas[kk][mm]*self.weightDecay)


    def standardizeTest(self, testX):
        standardizedTestX = (testX - self.XMean)/self.XStd
        return np.insert(standardizedTestX, 0, 1, axis=1)

    #assumes x is stadardized
    def predict(self, X):
        Z = np.insert(expit(np.dot(X, np.transpose(self.alphas))), 0, 1, axis=1)
        T = np.transpose(np.dot(self.betas, np.transpose(Z)))
        #predictedVal = softmax(T, axis=1)
        predictedVal = expit(T)
        results = np.zeros(len(X))
        bestArg = np.argmax(predictedVal, axis=1)
        for ii in range(len(results)):
            results[ii] = self.indicators[bestArg[ii]]
        return results