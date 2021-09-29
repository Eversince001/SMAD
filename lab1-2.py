import numpy as np
import random
from math import sqrt
import matplotlib.pyplot as plt

lvl_x1=4
lvl_x2=5
N = 25

def u(x1,x2):
    return 0.5 + 1 * x1 + 0.002 * x1**2 + 2 * x2 + 0.3 * x2**2

# f = [1, x1, x2, x1**2, x2**2]

#генерания факторов
def getFactors():
    X1=[]
    X2=[]
    for i in range (lvl_x1):
        x1 = random.uniform(-1, 1)
        X1.append(round(x1, 5))

    for i in range(lvl_x2):
        x2 = random.uniform(-1, 1)
        X2.append(round(x2, 5))

    for i in range(N-lvl_x1):
        x1 = random.randrange(0, 4)
        X1.append(X1[x1])

    for i in range(N-lvl_x2):
        x2 = random.randrange(0, 5)
        X2.append(X2[x2])


    return X1,X2

#среднее значение сигнала по выборке
def AverageSignalValue(U):
    AS_Value=0
    for i in U:
        AS_Value += i

    AS_Value /= len(U)
    return AS_Value

#мощность сигнала
def SignalPower(U):
    SP = 0
    AS_Value = AverageSignalValue(U)
    for i in U:
        SP += (i - AS_Value)**2

    SP = SP / (len(U) - 1)
    return SP

x1,x2=getFactors()

U=[]

for i in range(N):
    U.append(round(u(x1[i], x2[i]),5))

w2 = SignalPower(U)
sigma = sqrt(0.1*(w2))
y=[]
e=[]

for i in range(len(U)):
    e.append(round(np.random.normal(0, sigma),5))
    y.append(round((U[i] + e[i]),5))


X=[]

for i in range(N):
    X.append([0]*5)

for i in range(N):
    X[i][0] = 1

for i in range(N):
    X[i][1] = x1[i]
    X[i][2] = x2[i]
    X[i][3] = round(x1[i]**2, 5)
    X[i][4] = round(x2[i]**2, 5)


tettaR = []
Xt = np.matrix(X).transpose()
Xtemp = np.linalg.inv(Xt * np.matrix(X)) * Xt

for i in range(0, len(Xtemp)):
    temp = 0
    for j in range(0, np.size(Xtemp[i]) ):
        temp = temp + Xtemp[i,j] * y[j]
    tettaR.append(round(temp, 5))


y2 = []

for i in range(0, len(X)):
    temp = 0
    for j in range(0, np.size(X[i]) ):
        temp = temp + X[i][j] * tettaR[j]
    y2.append(round(temp, 5))

eR = np.array(y) - np.array(y2)

temp = 0
for i in range(len(eR)):
    temp += eR[i] * eR[i]
sigmaR = sqrt(temp / (N - len(tettaR)))

    
f = open("results.txt", 'w')
res = '(x1,\t x2)\t\t u\t\t\t e\t\t\t y\t\t\t y^\t\t\t y-y^\n'
f.write(res)
for i in range(len(y)):
    res = ''
    res += '('+ str(x1[i]) + ', ' + str(x2[i]) + '); '
    res += str(U[i]) + '; ' + str(e[i]) + '; ' + str(y[i]) + ';   '
    res += str(y2[i]) + ';   ' + str(round(eR[i], 5))
    res += '\n'
    f.write(res)
res += '\n\n'
res += "sigmaR = " + str(sigmaR) + '\n'
res += "F = " + str(sigmaR/sigma) + '\n'
res += "Tetta = " + str(tettaR)
f.write(res)   
f.close()


#График зависимости незашумленного отклика от фактора х1
fig = plt.figure
x1 = np.arange(-1,1, 0.1)
y = u(x1, 0)
plt.plot(x1,y)
#plt.show()


#График зависимости незашумленного отклика от фактора х2
fig = plt.figure
x2 = np.arange(-1,1, 0.1)
y = u(0, x2)
plt.plot(x2,y)
#plt.show()
