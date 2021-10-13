import numpy as np
import random
from scipy import stats
from math import sqrt

lvl_x1=4
lvl_x2=5
N = 30
alpha = 0.05

#Параметры исходной модели
tetta = [0.5, 1, 0.002, 2, 0.003, 0]

#Параметры измененной модели (добавлен новый регрессор x1*x2)
tettaChan = [0.5, 1, 0.002, 2, 0.003, 1]

def u_vec(x1, x2, t = tetta):
    x1 = np.array(x1)
    x2 = np.array(x2)
    f = [1, x1, x2, x1**2, x2**2, x1*x2]
    nu = np.dot(t, f)
    return nu

def u(x1, x2, t = tetta):
    f = [1, x1, x2, x1**2, x2**2, x1*x2]
    nu = 0
    for i in range (len(f)):
        nu = nu + t[i] * f[i]
    return nu

def Get_factors():
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

def Get_averageSignalValue(U):
    AS_Value=0
    for i in U:
        AS_Value += i

    AS_Value /= len(U)
    return AS_Value

def Get_signalPower(U):
    SP = 0
    AS_Value = Get_averageSignalValue(U)
    for i in U:
        SP += (i - AS_Value)**2
    SP = SP / (len(U) - 1)
    return SP

def Get_X():
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
    return X

def Get_tettaR():
    tettaR = []
    Xt = np.matrix(X).transpose()
    Xtemp = np.linalg.inv(Xt * np.matrix(X)) * Xt

    for i in range(0, len(Xtemp)):
        temp = 0
        for j in range(0, np.size(Xtemp[i]) ):
            temp = temp + Xtemp[i,j] * y[j]
        tettaR.append(round(temp, 5))
    return tettaR

def Get_y2():
    y2 = []
    for i in range(0, len(X)):
        temp = 0
        for j in range(0, np.size(X[i]) ):
            temp = temp + X[i][j] * tettaR[j]
        y2.append(round(temp, 5))
    return y2

def Get_U():
    U = []
    for i in range(N):
        U.append(round(u(x1[i], x2[i]),5))
    return U

def Get_sigma(U):
    w2 = Get_signalPower(U)
    return sqrt(0.1*(w2))

def Get_e():
    e = []
    U=Get_U()
    for i in range(len(U)):
        e.append(round(np.random.normal(0, sigma),5))
    return e

def Get_y():
    y = []
    U=Get_U()
    for i in range(len(U)):
        y.append(round((U[i] + e[i]),5))
    return y

def Get_sigmaR():
    temp = 0
    for i in range(len(eR)):
        temp += eR[i] * eR[i]
    return sqrt(temp / (N - len(tettaR)))

def SaveResultInFile():
    f = open("results.txt", 'w')
    res = '(x1,\t x2)\t\t u\t\t\t e\t\t\t y\t\t\t y^\t\t\t y-y^\n'
    f.write(res)
    for i in range(len(y)):
        res = ''
        res += '('+ str(x1[i]) + ', ' + str(x2[i]) + '); '
        res += str(U[i]) + '; ' + str(e[i]) + '; ' + str(y[i])
        res += '\n'
        f.write(res)
    f.close()

x1,x2 = Get_factors()
U = Get_U()
sigma = Get_sigma(U)
e = Get_e()
y = Get_y()

X = Get_X()
tettaR = Get_tettaR()

y2 = Get_y2()
eR = np.array(y) - np.array(y2)

sigmaR = Get_sigmaR()

print(sigma)
print(sigmaR)
print(sigmaR/sigma)
    
ty = stats.f.sf(1 - alpha, N - len(tetta), 99999999999999)
print(ty)

SaveResultInFile()