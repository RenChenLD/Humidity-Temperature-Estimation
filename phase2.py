import csv
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from phase1 import MODEL


class MODEL1(MODEL):
    def __init__(self, initialM, initialV):
        MODEL.__init__(self)
        self.iM = initialM
        self.iV = initialV
        self.beta = list()
        self.mean = list()
        self.variance = list()
        self.lr = LinearRegression()

    def train(self, trainData):
        train = [str.split(trainData[n], ",") for n in range(len(trainData))]
        train = train[1:]
        train = [m[1:] for m in train]
        for i in range(len(train)):
            for j in range(len(train[i])):
                train[i][j] = float(train[i][j])

        for i in range(len(train)):
            X = np.array(train[i][:-1])
            y = np.array(train[i][1:])
            # print X[i]
            tmpx = X.reshape(-1, 1)
            tmpy = y.reshape(-1, 1)

            self.lr.fit(tmpx, tmpy)
            self.beta.append([self.lr.intercept_[0], self.lr.coef_[0][0]])

    def nextMean(self, t, sensorID, result):
        nextM = self.beta[sensorID][0]
        nextM += self.beta[sensorID][1] * result[t - 1][sensorID]
        return round(nextM, 3)

    def nextVariance(self, t, sensorID, result):
        nextV = self.beta[sensorID][0]
        nextV += math.pow(self.beta[sensorID][1], 2) * result[t - 1][sensorID]
        return round(nextV, 3)

    def predict(self, outputPath, predictAppro, filePath, budget=0.5):
        train_01 = np.loadtxt(filePath, dtype=str)
        outputPath += str(int(budget * 100)) + ".csv"
        testData = [str.split(train_01[n], ",") for n in range(len(train_01))]
        for i in range(len(testData)):
            for j in range(1, len(testData[i])):
                testData[i][j] = round(float(testData[i][j]), 3)

        result = list()
        size = budget * 50
        size = int(size)

        if predictAppro == 'Window':
            selected = self.selectedWindow(size)
            predictAtT = self.iM
            for selectedNumber in selected[0]:
                predictAtT[selectedNumber - 1] = testData[selectedNumber][1]
            result.append(predictAtT)
            for t in range(1, 96):
                predictAtT = []
                for s in range(0, 50):
                    predictAtT.append(self.nextMean(t, s, result))
                for selectedNumber in selected[t]:
                    predictAtT[selectedNumber - 1] = testData[selectedNumber][t + 1]
                result.append(predictAtT)

        elif predictAppro == 'Variance':
            selected = self.selectedVar(size, self.iV)
            preVar = []
            preVar.append(self.iV)
            predictAtT = self.iM
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
                    predictAtT[selectedNumber - 1] = testData[selectedNumber][t + 1]
                    tmpV[selectedNumber - 1] = 0
                result.append(predictAtT)
                preVar.append(tmpV)
        result = self.reformatResult(result, testData)
        with open(outputPath, 'wb') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerows(result)

    def selectedVar(self, size, lis):
        tmplis = sorted(lis, reverse=True)
        resultLis = list()
        for item in tmplis[:size]:
            tmp = lis.index(item)
            while resultLis.__contains__(tmp + 1):
                nextl = lis[tmp + 1:]
                tmp += nextl.index(item) + 1
            resultLis.append(lis.index(item) + 1)
        return resultLis

    def reformatResult(self, result, data):
        tmp = list()
        tmp.append(data[0])
        for i in range(1, len(data)):
            tmp.append([data[i][0]])
        for t in range(96):
            for j in range(len(result[t])):
                tmp[j + 1].append(format(result[t][j], '.2E'))
                # tmp[j+1].append(result[t][j])
        return tmp


class MODEL2(MODEL):
    def __init__(self, initialM, initialV):
        MODEL.__init__(self)
        self.iM = initialM
        self.iV = initialV
        self.beta = list()
        self.mean = list()
        self.variance = list()
        self.lr = LinearRegression()

    def nextMean(self, t, sensorID, result):
        if t < 48:
            ti = t
        else:
            ti = t - 48
        nextM = self.beta[sensorID][ti][0]
        nextM += self.beta[sensorID][ti][1] * result[t - 1][sensorID]
        nextM = round(nextM, 3)
        return nextM

    def nextVariance(self, t, sensorID, result):
        if t <= 48:
            ti = t
        else:
            ti = t - 49
        nextV = self.beta[sensorID][ti-1][0]
        nextV += math.pow(self.beta[sensorID][ti-1][1], 2) * result[t - 1][sensorID]
        return nextV

    def train(self, trainData):
        train = [str.split(trainData[n], ",") for n in range(len(trainData))]
        for i in range(len(train)):
            for j in range(1, len(train[i])):
                train[i][j] = float(train[i][j])
        train = train[1:]
        train = [m[1:] for m in train]

        for i in range(len(train)):
            tmp = list()
            for j in range(0, 47):
                list_temp = []
                list_temp.append(train[i][j])
                list_temp.append(train[i][j + 48])
                list_temp.append(train[i][j + 96])
                y = [train[i][j + 1], train[i][j + 1 + 48], train[i][j + 1 + 96]]
                X = np.asarray(list_temp)
                X = np.reshape(X, (-1, 1))
                y = np.reshape(y, (-1, 1))
                self.lr.fit(X, y)
                tmp.append([round(self.lr.intercept_[0], 3), round(self.lr.coef_[0][0], 3)])
            list_temp = []
            list_temp.append(train[i][47])
            list_temp.append(train[i][95])
            y = [train[i][48], train[i][96]]
            X = np.asarray(list_temp)
            X = np.reshape(X, (-1, 1))
            y = np.reshape(y, (-1, 1))
            self.lr.fit(X, y)
            tmp.append([round(self.lr.intercept_[0], 3), round(self.lr.coef_[0][0], 3)])
            self.beta.append(tmp)

    def predict(self, outputPath, predictAppro, filePath, budget=0.5):
        train_01 = np.loadtxt(filePath, dtype=str)
        outputPath += str(int(budget * 100)) + ".csv"
        testData = [str.split(train_01[n], ",") for n in range(len(train_01))]
        for i in range(len(testData)):
            for j in range(1, len(testData[i])):
                testData[i][j] = float(testData[i][j])

        result = list()
        size = budget * 50
        size = int(size)

        if predictAppro == 'Window':
            selected = self.selectedWindow(size)
            predictAtT = self.iM
            for selectedNumber in selected[0]:
                predictAtT[selectedNumber - 1] = testData[selectedNumber][1]
            result.append(predictAtT)
            for t in range(1, 96):
                predictAtT = []
                for s in range(0, 50):
                    predictAtT.append(self.nextMean(t, s, result))
                for selectedNumber in selected[t]:
                    predictAtT[selectedNumber - 1] = round(testData[selectedNumber][t + 1], 3)
                result.append(predictAtT)

        elif predictAppro == 'Variance':
            preVar = [self.iV]
            predictAtT = self.iM
            selected = self.selectedVar(size, preVar)
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
                    predictAtT[selectedNumber - 1] = testData[selectedNumber][t + 1]
                    tmpV[selectedNumber - 1] = 0
                result.append(predictAtT)
                preVar.append(tmpV)
        result = self.reformatResult(result, testData)
        with open(outputPath, 'wb') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerows(result)

    def selectedVar(self, size, lis):
        tmplis = sorted(lis, reverse=True)
        resultLis = list()
        for item in tmplis[:size]:
            tmp = lis.index(item)
            while resultLis.__contains__(tmp + 1):
                nextl = lis[tmp + 1:]
                tmp += nextl.index(item) + 1
            resultLis.append(lis.index(item) + 1)
        return resultLis

    def reformatResult(self, result, data):
        tmp = list()
        tmp.append(data[0])
        for i in range(1, len(data)):
            tmp.append([data[i][0]])
        for t in range(96):
            for j in range(len(result[t])):
                tmp[j + 1].append(format(result[t][j], '.2E'))
                # tmp[j+1].append(result[t][j])
        return tmp


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
    train_data_T = np.loadtxt('intelTemperatureTrain.csv', 'str')

    budgets = [0, 0.05, 0.1, 0.2, 0.25]

    # initialMH, initialVH = calInitial(train_data_H)
    # m = MODEL2(initialMH, initialVH)
    # m.train(train_data_H)
    # m.predict('Humidity/d-v', 'Variance', 'intelHumidityTest.csv', budget=budgets[4])

    for budget in budgets:
        initialMH, initialVH = calInitial(train_data_H)
        m = MODEL2(initialMH, initialVH)
        m.train(train_data_H)
        m.predict('Humidity/d-w', 'Window', 'intelHumidityTest.csv', budget=budget)

    for budget in budgets:
        initialMH, initialVH = calInitial(train_data_H)
        m = MODEL2(initialMH, initialVH)
        m.train(train_data_H)
        m.predict('Humidity/d-v', 'Variance', 'intelHumidityTest.csv', budget=budget)

    for budget in budgets:
        initialMT, initialVT = calInitial(train_data_T)
        n = MODEL2(initialMT, initialVT)
        n.train(train_data_T)
        n.predict('Temperature/d-w', 'Window', 'intelTemperatureTest.csv', budget=budget)

    for budget in budgets:
        initialMT, initialVT = calInitial(train_data_T)
        n = MODEL2(initialMT, initialVT)
        n.train(train_data_T)
        n.predict('Temperature/d-v', 'Variance', 'intelTemperatureTest.csv', budget=budget)
