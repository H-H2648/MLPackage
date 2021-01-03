import numpy as np
import math
from scipy import linalg
from scipy.special import expit

#it kinda works (inconsistet with different starting weights)
#not sure how to fix that

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
            self.betas = np.random.uniform(low=-0.7, high = 0.7, size = ((self.K, self.M)))
        else:
            self.betas = np.random.uniform(low = -0.7, high = 0.7, size=self.M)

    def feed(self, learningRate):
        X = self.inputArray
        Z = expit(np.dot(X, np.transpose(self.alphas)))
        T = np.dot(self.betas, np.transpose(Z))
        predictedVal = T
        if self.outputArray.ndim > 1:
            for mm in range(self.M):
                for ll in range(self.p + 1):
                    dRa = np.zeros(len(self.inputArray))
                    for kk in range(self.K):
                        dRa -= 2*np.multiply((self.outputArray[kk] - predictedVal[kk])*self.betas[kk][mm], np.multiply(np.multiply(expit(np.dot(X, self.alphas[mm])),(1 - expit(np.dot(X, self.alphas[mm])))), X[:,ll]))
                    self.alphas[mm][ll] -= learningRate*(np.average(dRa) +2*self.alphas[mm][ll]*self.weightDecay)
            for kk in range(self.K):
                for mm in range(self.M):
                    dRB = np.zeros(len(self.inputArray))
                    dRB -= np.multiply(2*(self.outputArray[kk] - predictedVal[kk]), Z[:,mm])
                    self.betas[kk][mm] -= learningRate*(np.average(dRB) + 2*self.betas[kk][mm]*self.weightDecay)
        else:
            dAs = np.zeros((self.M, self.p+1))
            dBs = np.zeros(self.M)
            difference = self.outputArray - predictedVal
            for mm in range(self.M):
                for ll in range(self.p + 1):
                    derivG = 1
                    sigma = expit(np.dot(X, self.alphas[mm]))
                    derivSigma = np.multiply(sigma, 1- sigma)
                    dRa =  np.multiply(-2*difference*derivG*self.betas[mm], np.multiply(derivSigma, X[:,ll]))
                    dAs[mm][ll] = np.sum(dRa)
                    self.alphas[mm][ll] -= learningRate*(np.average(dRa) + 2*self.alphas[mm][ll]*self.weightDecay)
            for mm in range(self.M):
                dRB = -np.multiply(2*difference, Z[:,mm])
                dBs[mm] = np.sum(dRB)
                self.betas[mm] -= learningRate*(np.average(dRB) + 2*self.betas[mm] * self.weightDecay)
        print('Alpha Derivatives:')
        print(linalg.norm(dAs))
        print('Beta Derivatives:')
        print(linalg.norm(dBs))
        print('Current Norms')
        print(1/2 * linalg.norm(difference)**2/len(self.inputArray))

    def standardizeTest(self, testX):
        standardizedTestX = (testX - self.XMean)/self.XStd
        return np.insert(standardizedTestX, 0, 1, axis=1)

#assumes x is stadardized
    def predict(self, x):
        z = expit(np.dot(self.alphas, x))
        t = np.dot(self.betas, z)
        predictedVal = t
        return predictedVal
        



