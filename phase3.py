import csv
import numpy as np
import math
from sklearn.linear_model import Lasso
from phase2 import MODEL1



class MODEL3(MODEL1):
    def __init__(self, im, iv):
        MODEL1.__init__(self, im, iv)
        self.lasso = Lasso()
        self.lasso.max_iter = 2000

    def calculateBeta(self, trainData):
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
            # self.lr.fit(X, y)
            # tmp.append(self.lr.intercept_)
            # for a in self.lr.coef_:
            self.lasso.fit(X, y)
            tmp.append(self.lasso.intercept_)
            for a in self.lasso.coef_:
                tmp.append(a)
            beta.append(tmp)
        betas = beta
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
        self.beta = self.calculateBeta(trainData)

    def predict(self, outputPath, predictAppro, filePath, budget=0.5):
        size = int(budget * 50)
        outputPath += str(int(budget * 100)) + ".csv"
        result = []
        train_01 = np.loadtxt(filePath, dtype=str)
        testData = [str.split(train_01[n], ",") for n in range(len(train_01))]
        for i in range(len(testData)):
            for j in range(1, len(testData[i])):
                testData[i][j] = float(testData[i][j])
        if predictAppro == 'Window':
            selected = self.selectedWindow(size)
            predictAtT = iM
            for selectedNumber in selected[0]:
                predictAtT[selectedNumber - 1] = testData[selectedNumber][1]
            result.append(predictAtT)
            for t in range(1, 96):
                predictAtT = []
                for s in range(0, 50):
                    predictAtT.append(self.nextMean(t, s, result))
                for selectedNumber in selected[t]:
                    predictAtT[selectedNumber - 1] = testData[selectedNumber][t]
                result.append(predictAtT)
        elif predictAppro == 'Variance':
            selected = self.selectedVar(size, iV)
            preVar = []
            preVar.append(iV)
            predictAtT = iM
            for selectedNumber in selected:
                predictAtT[selectedNumber - 1] = testData[selectedNumber][1]
                preVar[0][selectedNumber - 1] = 0
            result.append(predictAtT)
            for t in range(1, 96):
                predictAtT = []
                tmpV = []
                for s in range(50):
                    predictAtT.append(self.nextMean(t, s, result))
                    tmpV.append(self.nextVariance(t, s, preVar))
                selected = self.selectedVar(size, tmpV)
                for selectedNumber in selected:
                    predictAtT[selectedNumber - 1] = testData[selectedNumber][t]
                    tmpV[selectedNumber - 1] = 0
                result.append(predictAtT)
                preVar.append(tmpV)
        result = self.reformatResult(result, testData)
        with open(outputPath, 'wb') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerows(result)
        print "Results have been exported to file: " + outputPath


def calInitial(trainData):
    data = [str.split(trainData[n], ",") for n in range(len(trainData))]
    initial_mean = list()
    initial_var = list()
    data = [m[1:] for m in data]
    data = data[1:]
    for i in range(len(data)):
        tmpM = 0
        for j in range(1, len(data[i])):
            data[i][j] = float(data[i][j])
            tmpM += data[i][j]
        tmpM /= len(data[i])
        initial_mean.append(round(tmpM, 3))
    for i in range(len(data)):
        tmpV = 0
        for j in range(1, len(data[i])):
            tmpV += math.pow(data[i][j] - initial_mean[i], 2)
        initial_var.append(round(tmpV, 3))
    return initial_mean, initial_var


if __name__ == '__main__':
    train_data_H = np.loadtxt('intelHumidityTrain.csv', 'str')
    iM, iV = calInitial(train_data_H)
    # print iM, iV
    # print len(iM)
    # m = MODEL3(iM, iV)
    # m.train(train_data_H)

    budgets = [0, 0.05, 0.1, 0.2, 0.25]
    # m.predict('Humidity/w-', 'Window', 'intelHumidityTest.csv', budget=budgets[1])
    for budget in budgets:
        m = MODEL3(iM, iV)
        m.train(train_data_H)
        m.predict('Humidity/w', 'Window', 'intelHumidityTest.csv', budget=budget)
        m.predict('Humidity/v', 'Variance', 'intelHumidityTest.csv', budget=budget)

    train_data_T = np.loadtxt('intelTemperatureTrain.csv', 'str')
    iM, iV = calInitial(train_data_T)

    for budget in budgets:
        n = MODEL3(iM, iV)
        n.train(train_data_T)
        n.predict('Temperature/w', 'Window', 'intelTemperatureTest.csv', budget=budget)
        n.predict('Temperature/v', 'Variance', 'intelTemperatureTest.csv', budget=budget)
