import numpy as np
import csv
import math


class MODEL:
    def __init__(self):
        self.mean = list()
        self.variance = list()

    def calculate_mean(self, filePath):
        train_01 = np.loadtxt(filePath, dtype=str)
        data = [str.split(train_01[n], ",") for n in range(len(train_01))]
        # self.mean.append(data[0][0:49])

        for i in range(1,len(data)):
            for j in range(1, len(data[i])):
                data[i][j] = float(data[i][j])
        # print data
        for i in range(1, len(data)):
            tmpMean = list()
            # tmpMean.append(data[i][0])
            for j in range(1, 49):
                tmp = (data[i][j] + data[i][j + 48] + data[i][j + 96]) / 3.00
                # tmp = format(tmp, '.2E')
                tmpMean.append(round(tmp, 3))
            self.mean.append(tmpMean)

    def calculate_variance(self, filePath):
        train_01 = np.loadtxt(filePath, dtype=str)
        data = [str.split(train_01[n], ",") for n in range(len(train_01))]

        # self.variance.append(data[0][0:49])

        for i in range(len(data)):
            for j in range(1, len(data[i])):
                data[i][j] = float(data[i][j])

        for i in range(1, len(data)):
            tmpVar = list()
            # tmpVar.append(data[i][0])
            for j in range(1, 49):
                tmp = (data[i][j] + data[i][j + 48] + data[i][j + 96]) / 3.00
                tmp2 = math.pow((data[i][j] - tmp), 2) + math.pow((data[i][j + 48] - tmp), 2) + math.pow(
                    (data[i][j + 96] - tmp), 2)
                tmp2 /= 3.00
                # tmp2 = format(tmp2, '.2E')
                tmpVar.append(round(tmp2, 3))
            self.variance.append(tmpVar)

    def selectedWindow(self, size):
        total = list()
        s = 1
        for t in range(1, 97):

            l = list()

            if s + size > 50:
                for i in range(s, 51):
                    l.append(i)
                s = s + size - 50
                for i in range(1, s):
                    l.append(i)
            else:
                for i in range(s, s + size):
                    l.append(i)
                s = s + size
            total.append(l)

        return total

    def selectedVar(self, size):
        lis = self.variance
        returnList = list()
        for i in range(len(self.variance)):
            tmplis = sorted(lis[i], reverse=True)
            resultLis = list()
            for item in tmplis[:size]:
                tmp = lis[i].index(item)
                while resultLis.__contains__(tmp+1):
                    nextl = lis[i][tmp+1:]
                    tmp += nextl.index(item) +1
                resultLis.append(tmp +1)

            returnList.append(resultLis)
        return returnList

    def predict(self, predictType, predictAppro, filePath, budget=0.5):
        """

        :param filePath:
        :type budget: float
        """
        train_01 = np.loadtxt(filePath, dtype=str)
        data = [str.split(train_01[n], ",") for n in range(len(train_01))]
        for i in range(len(data)):
            for j in range(1, len(data[i])):
                data[i][j] = float(data[i][j])

        predict = list()
        predict.append(data[0][0:97])
        size = int(50 * budget)
        t = 0

        if predictAppro == 'Window':
            for i in range(len(self.mean)):
                tmp = list()
                tmp.append(data[i+1][0])
                for j in range(len(self.mean[i])):
                    tmp.append(format(self.mean[i][j], '.2E'))
                for j in range(len(self.mean[i])):
                    tmp.append(format(self.mean[i][j], '.2E'))
                predict.append(tmp)
            selectedSensors = self.selectedWindow(size)
            for i in range(len(selectedSensors)):
                for j in range(len(selectedSensors[i])):
                    tmp = selectedSensors[i][j]
                    predict[tmp][i+1] = format(data[tmp][i+1], '.2E')
        else:
            for i in range(len(self.mean)):
                tmp = list()
                tmp.append(data[i+1][0])
                for j in range(len(self.mean[i])):
                    tmp.append(format(self.mean[i][j], '.2E'))
                for j in range(len(self.mean[i])):
                    tmp.append(format(self.mean[i][j], '.2E'))
                predict.append(tmp)
            selectedSensors = self.selectedVar(size)
            print selectedSensors
            for i in range(len(selectedSensors)):
                for j in range(len(selectedSensors[i])):
                    tmp = selectedSensors[i][j]
                    predict[tmp][i + 1] = format(data[tmp][i + 1], '.2E')
        # print predict
        with open(predictType, 'wb') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerows(predict)


if __name__ == '__main__':
    m = MODEL()
    n = MODEL()
    m.calculate_mean('intelHumidityTrain.csv')
    n.calculate_mean('intelTemperatureTrain.csv')
    m.calculate_variance('intelHumidityTrain.csv')
    n.calculate_variance('intelTemperatureTrain.csv')

    # print len(m.mean), len(m.mean[0])
    # print len(m.variance), len(m.mean[0])

    budgets = [0, 0.05, 0.1, 0.2, 0.25]
    # m.predict('Humidity/w' + '5' + '.csv', 'Window', 'intelHumidityTest.csv', budget=budgets[1])
    for budget in budgets:
        num = str(int(budget*100))
        m.predict('Humidity/w' + num + '.csv', 'Window', 'intelHumidityTest.csv', budget=budget)
        m.predict('Humidity/v' + num + '.csv', 'Variance', 'intelHumidityTest.csv', budget=budget)
        n.predict('Temperature/w' + num + '.csv', 'Window', 'intelTemperatureTest.csv', budget=budget)
        n.predict('Temperature/v' + num + '.csv', 'Variance', 'intelTemperatureTest.csv', budget=budget)
