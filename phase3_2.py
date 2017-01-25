from phase2 import MODEL1
import numpy as np
import math
from sklearn.linear_model import Lasso


class MODEL3_2(MODEL1):
    def __init__(self,iM, iV):
        MODEL1.__init__(self,initialM=iM, initialV=iV)
        self.lasso = Lasso()
        self.lasso.max_iter = 2000

    def calculate_betas(self, trainData):
        data = [str.split(trainData[n], ",") for n in range(len(trainData))]
        data = [m[1:] for m in data]
        data = data[1:]
        tmpa = list()
        for i in range(len(data[0])):
            tmpb = list()
            for j in range(len(data)):
                tmpb.append(float(data[j][i]))
            tmpa.append(tmpb)

        X = np.array([tmpa[i][:] for i in range(len(tmpa) - 1)])
        beta = list()
        for i in range(len(data)):
            tmp = list()
            y = [float(m) for m in data[i]]
            y = y[1:]
            self.lasso.fit(X, y)
            tmp.append(self.lasso.intercept_)
            for a in self.lasso.coef_:
                tmp.append(a)
            beta.append(tmp)
        betas = beta
        print len(betas), len(betas[0])
        return betas

    def nextMean(self, t, sensorID, result):
        nextM = self.beta[sensorID][0]
        for i in range(len(result[t - 1])):
            nextM += self.beta[sensorID][i + 1] * result[t - 1][i]
        nextM = round(nextM, 3)
        return nextM

    def nextVariance(self, t, sensorID, result):
        nextV = self.beta[sensorID][0]
        for i in range(len(result[t - 1])):
            nextV += math.pow(self.beta[sensorID][i + 1], 2) * result[t - 1][i]
        return nextV

    def train(self, trainData):
        self.beta = self.calculate_betas(trainData)

    def predict(self, outputPath, predictAppro, filePath, budget=0.5):
        size = int(budget * 50)
        outputPath += str(int(budget * 100)) + ".csv"
        result = []
        train_01 = np.loadtxt(filePath, dtype=str)
        testData = [str.split(train_01[n], ",") for n in range(len(train_01))]
        for i in range(len(testData)):
            for j in range(1, len(testData[i])):
                testData[i][j] = float(testData[i][j])

        predict = self.iM
        result.append(predict)
        for t in range(1, 96):
            predictAtT = []
            for s in range(0, 50):
                predictAtT.append(self.nextMean(t, s, result))
            result.append(predictAtT)

        # if predictAppro == 'Window'and budget != 0:
        #     selected = self.selectedWindow(size)
        #     for t in range(96):
        #         for selectedNumber in selected[t]:
        #
        # elif predictAppro == 'Variance' and budget != 0:
        #     selected = self.selectedVar(size, self.iV)
        #