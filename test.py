import numpy as np
import math
from sklearn.linear_model import LinearRegression
from  phase3_2 import MODEL3_2


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
    trainData = np.loadtxt("intelHumidityTrain.csv", dtype='str')

    # iM, iV = calInitial(trainData)
    # m = MODEL3_2(iM, iV)
    # m.train(trainData)
    # m.predict('Humidity/w-', 'Window', 'intelHumidityTest.csv', budget=0.25)

    a = [1,1]
    print a.index(1)
    print

