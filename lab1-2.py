import numpy as np
import random
from scipy.linalg import *
from math import sqrt
from matplotlib import pyplot as plt

lvl_x1=4
lvl_x2=5
N = 30

def u(x1,x2):
    return 0.5 + 1 * x1 + 0.002 * x1**2 + 2 * x2 + 0.003 * x2**2

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

def AverageSignalValue(U):
    AS_Value=0
    for i in U:
        AS_Value += i

    AS_Value /= len(U)
    return AS_Value

def SignalPower(U):
    TrueValues = 0
    AS_Value = AverageSignalValue(U)
    for i in U:
        TrueValues += (i - AS_Value)**2

    TrueValues = TrueValues / (len(U) - 1)
    return TrueValues

x1,x2=getFactors()
U=[]

for i in range(N):
    U.append(round(u(x1[i], x2[i]),5))


w2 = SignalPower(U)
sigma = sqrt(0.15*(w2))
y=[]
e=[]

for i in range(len(U)):
    e.append(round(np.random.normal(0, sigma),5))
    y.append(round((U[i] + e[i]),5))

f = open("results.txt", 'w')
res = '(x1,\t x2)\t\t u\t\t\t e\t\t\t y\n'
f.write(res)
for i in range(len(y)):
    res = ''
    res += '('+ str(x1[i]) + ', ' + str(x2[i]) + '); '
    res += str(U[i]) + '; ' + str(e[i]) + '; ' + str(y[i])
    res += '\n'
    f.write(res)
f.close()