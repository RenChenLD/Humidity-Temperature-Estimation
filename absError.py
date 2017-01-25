import numpy as np
def Hu(HuTest):
    w = list()
    v = list()
    num = [0,5,10,20,25]
    for i in num:
        tmp_data = np.loadtxt('Humidity/w' + str(i) + '.csv', dtype=str)
        tmpw = [str.split(tmp_data[n], ",") for n in range(len(tmp_data))]
        w.append(tmpw)
        tmp_data = np.loadtxt('Humidity/v' + str(i) + '.csv', dtype=str)
        tmpv = [str.split(tmp_data[n], ",") for n in range(len(tmp_data))]
        v.append(tmpv)

    for i in range(1, len(HuTest)):
        for j in range(1, len(HuTest[i])):
            for n in range(5):
                w[n][i][j] = float(w[n][i][j])
                v[n][i][j] = float(v[n][i][j])
    n =[0,0,0,0,0]
    for i in range(1,len(TeTest)):
        for j in range(1, len(TeTest[i])):
            for k in range(5):
                n[k] += abs(HuTest[i][j]-w[k][i][j])
    for i in n:
        print i/4800

    n =[0,0,0,0,0]
    for i in range(1,len(TeTest)):
        for j in range(1, len(TeTest[i])):
            for k in range(5):
                n[k] += abs(HuTest[i][j]-v[k][i][j])
    for i in n:
        print i/4800

def Te(TeTest):
    w = list()
    v = list()
    num = [0,5,10,20,25]
    for i in num:
       tmp_data = np.loadtxt('Temperature/w' + str(i) + '.csv', dtype=str)
       tmpw = [str.split(tmp_data[n], ",") for n in range(len(tmp_data))]
       w.append(tmpw)
       tmp_data = np.loadtxt('Temperature/v' + str(i) + '.csv', dtype=str)
       tmpv = [str.split(tmp_data[n], ",") for n in range(len(tmp_data))]
       v.append(tmpv)
    for i in range(1, len(TeTest)):
        for j in range(1, len(TeTest[i])):
            for n in range(5):
                w[n][i][j] = float(w[n][i][j])
                v[n][i][j] = float(v[n][i][j])

    n =[0,0,0,0,0]
    for i in range(1,len(TeTest)):
        for j in range(1, len(TeTest[i])):
            for k in range(5):
                n[k] += abs(TeTest[i][j]-w[k][i][j])
    for i in n:
        print i/4800

    n =[0,0,0,0,0]
    for i in range(1,len(TeTest)):
        for j in range(1, len(TeTest[i])):
            for k in range(5):
                n[k] += abs(TeTest[i][j]-v[k][i][j])
    for i in n:
        print i/4800

if __name__ == '__main__':
    tmp_data = np.loadtxt("intelHumidityTest.csv", dtype=str)
    HuTest = [str.split(tmp_data[n], ",") for n in range(len(tmp_data))]
    tmp_data = np.loadtxt("intelTemperatureTest.csv", dtype=str)
    TeTest = [str.split(tmp_data[n], ",") for n in range(len(tmp_data))]

    for i in range(1, len(HuTest)):
        for j in range(1, len(HuTest[i])):
            HuTest[i][j] = float(HuTest[i][j])
            TeTest[i][j] = float(TeTest[i][j])

    print 'Humidity Absolute Error:'
    Hu(HuTest)

    print '\n'

    print  'Temperature Absolute Error:'
    Te(TeTest)

