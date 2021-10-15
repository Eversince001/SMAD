import numpy as np
import random
from scipy import stats
from math import sqrt
import csv
import pandas as pd
import matplotlib.pyplot as plot
from scipy.stats import t as Student
from scipy.stats import f as Fisher

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

def Get_X(x1, x2):
    X=[]

    #x1 = np.array(x1)
    #x2 = np.array(x2)
    #f = [1, x1, x2, x1**2, x2**2, x1*x2]
    #for i in range(N):
    #    X.append(f[i])

    for i in range(N):
        X.append([0]*len(tetta))

    for i in range(N):
        X[i][0] = 1

    for i in range(N):
        X[i][1] = x1[i]
        X[i][2] = x2[i]
        X[i][3] = round(x1[i]**2, 5)
        X[i][4] = round(x2[i]**2, 5)
        X[i][5] = round(x1[i] * x2[i], 5)
    return X

def Get_tettaR(X, y):
    tettaR = []
    Xt = np.matrix(X).transpose()
    Xtemp = np.linalg.inv(Xt * np.matrix(X)) * Xt

    for i in range(0, len(Xtemp)):
        temp = 0
        for j in range(0, np.size(Xtemp[i]) ):
            temp = temp + Xtemp[i,j] * y[j]
        tettaR.append(round(temp, 5))
    return tettaR

def Get_y2(X, tettaR):
    y2 = []
    for i in range(0, len(X)):
        temp = 0
        for j in range(0, np.size(X[i]) ):
            temp = temp + X[i][j] * tettaR[j]
        y2.append(round(temp, 5))
    return y2

def Get_U(x1,x2):
    U = []
    for i in range(N):
        U.append(round(u(x1[i], x2[i]),5))
    return U

def Get_sigma(U):
    w2 = Get_signalPower(U)
    return sqrt(0.1*(w2))

def Get_e(U, sigma):
    e = []
    for i in range(len(U)):
        e.append(round(np.random.normal(0, sigma),5))
    return e

def Get_y(U, e):
    y = []
    for i in range(len(U)):
        y.append(round((U[i] + e[i]),5))
    return y

def Get_sigmaR(eR, tettaR):
    temp = 0
    for i in range(len(eR)):
        temp += eR[i] * eR[i]
    return sqrt(temp / (N - len(tettaR)))

def Get_djj(X):
    Xtemp = np.linalg.inv(np.dot(np.matrix(X).transpose(), np.matrix(X)))
    djjj = []
    for i in range(len(Xtemp)):
        djjj.append(Xtemp[i,i])
    return djjj

def GetConfidenceInterval(tettaR, Ft, sigmaR, djj):
    D_upper = np.zeros(len(tettaR))
    D_lower = np.zeros(len(tettaR))
    for i in range(len(tettaR)):
        D_upper[i] = round(tettaR[i] - Ft * np.sqrt(sigmaR * djj[i]), 5)
        D_lower[i] = round(tettaR[i] + Ft * np.sqrt(sigmaR * djj[i]), 5)
    return D_upper, D_lower

def Get_StatF(tettaR, djj, sigmaR):
    Fstat = []
    for i in range(len(tettaR)):
        Fstat.append(round((tettaR[i]) ** 2 / (sigmaR * djj[i]), 5))
    return Fstat

def Get_RRS(y, X, tettaR):
    RSS = np.dot(np.transpose(y - np.dot(X, tettaR)), y - np.dot(X, tettaR))
    return RSS

def Get_RRSH(y):
    RSSH = 0
    for i in range(len(y)): 
        RSSH += (y[i] - np.average(y))**2
    return RSSH

def Get_valHip(RSS, RSSH, N, m):
    return ((RSSH - RSS)/ (m - 1)) / (RSS / (N - m))

f = lambda x1, x2: [1, x1, x2, x1**2, x2**2, x1*x2]
def Get_confidenceIntervalForMOandRespond(x1, x2, uM, sigma, sigmaR, tettaR, X):
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    M_upper=[]
    M_lower=[]
    respond_upper=[]
    respond_lower=[]
    u_dov = u_vec(x1, x2, tettaR)
    Fs = Student.ppf(1 - alpha/2, N - len(tettaR))

    for i in range(len(uM)):
        Xtemp = np.linalg.inv(np.matrix(X).transpose() * np.matrix(X))
        fx = np.array(f(x1[i], 0))
        #fx = np.array(f(0, x2[i]))
        vkl1 = np.matmul(fx.transpose(), Xtemp) 
        vkl2 = vkl1@fx.T
        vkl = sigma * float(np.sqrt(vkl2))
        M_upper.append(u_dov[i] - (Fs * vkl)) 
        M_lower.append(u_dov[i] + (Fs * vkl))

        vkl_otkl = float(sigmaR * (1 + vkl2))
        respond_lower.append(u_dov[i] - (Fs * vkl_otkl)) 
        respond_upper.append(u_dov[i] + (Fs * vkl_otkl))

    return M_upper, M_lower, respond_lower, respond_upper, u_dov

def create_plot(data1, data2, data3, x, t=0): 
    fig, ax = plot.subplots() 
    if (t == 0):
        ax.set_title("Доверительный интервал для математичсекого ожидания.") 
    else:
        ax.set_title("Доверительный интервал для отклика.") 
    ax.plot(x, data1, label='Левая граница') 
    ax.plot(x, data2, label='Теоретическое значение') 
    ax.plot(x, data3, label='Правая граница') 
    ax.set_ylabel("у") 
    ax.set_xlabel("х") 
    ax.legend() 
    ax.grid() 
    fig.set_figheight(5) 
    fig.set_figwidth(16) 
    plot.show() 
    plot.show()

def SaveDataInTextFile(x1, x2, y, e, U):
    f = open("data.txt", 'w')
    res = '(x1,\t x2)\t\t u\t\t\t e\t\t\t y\t\t\t y^\t\t\t y-y^\n'
    f.write(res)
    for i in range(len(y)):
        res = ''
        res += '('+ str(x1[i]) + ', ' + str(x2[i]) + '); '
        res += str(U[i]) + '; ' + str(e[i]) + '; ' + str(y[i])
        res += '\n'
        f.write(res)
    f.close()

def SaveDataInCSVFile(x1, x2, y, e, U):
    data = dict(col1=x1, col2=x2, col3=y, col4=e, col5=U)
    df = pd.DataFrame(data)
    df.to_csv(r'data.csv', sep=';', header=False, index=False)

def ReadDataFromCSVFile():
        with open('data.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            dataForAnalys = np.array(list(csv_reader))
            np.delete(dataForAnalys, (0), axis=0)
            x1 = dataForAnalys[:, 0]
            x1 = [float(x) for x in x1]
            x2 = dataForAnalys[:, 1]
            x2 = [float(x) for x in x2]
            y = dataForAnalys[:, 2]
            y = [float(y) for y in y]
            e = dataForAnalys[:, 3]
            e = [float(e) for e in e]
            U = dataForAnalys[:, 4]
            U = [float(U) for U in U]
            return x1, x2, e, y, U

def main():
#lw1
    #x1,x2 = Get_factors()
    #U = Get_U(x1,x2)
    x1, x2, e, y, U = ReadDataFromCSVFile()
    sigma = Get_sigma(U)
    #e = Get_e(U, sigma)
    #y = Get_y(U, e)
    
#lw2
    f = open("All results.txt", 'w')
    X = Get_X(x1, x2)
    tettaR = Get_tettaR(X, y)

    y2 = Get_y2(X, tettaR)
    eR = np.array(y) - np.array(y2)

    sigmaR = Get_sigmaR(eR, tettaR)
    F = sigmaR/sigma

    f.write("tetta = " + str(tetta) + '\n')
    f.write("tettaR = " + str(tettaR) + '\n')
    f.write("sigma = " + str(round(sigma, 5)) + '\n')
    f.write("sigmaR = " + str(round(sigmaR, 5)) + '\n')
    f.write("F = " + str(round(sigmaR/sigma, 5)) + '\n\n\n')
    
#lw3
    #lw3:2 Доверительный интервал
    Ft = stats.t.ppf(1 - alpha, N - len(tetta))
    djj = Get_djj(X)
    D_upper, D_lower = GetConfidenceInterval(tettaR, Ft, sigmaR, djj)
    f.write('Confidence interval\n')
    f.write('Lower limit\t\ttetta\t\tThetaR\t\t\tUpper limit' + '\n')
    for i in range(len(tettaR)):
        f.write(str(D_lower[i]) + '\t\t\t' + str(tetta[i]) + '\t\t\t' + str(tettaR[i]) + '\t\t\t' + str(D_upper[i]) + '\n')


    #lw3:3 Вычисление гипотезы о незначимости параметров
    Fstat = Get_StatF(tettaR, djj, sigmaR)
    Ff = stats.f.ppf(1-alpha/2, 1, N - len(tetta))
    resultEx3 = []
    for i in range(len(tettaR)):
        if(Fstat[i] < Ff):
            resultEx3.append('+')
        else:
            resultEx3.append('-')

    f.write('\n\nF\t\t\tParameter estimation\t\tResult' + '\n')
    for i in range(len(tettaR)):
        f.write(str(Fstat[i]) + '\t\t' + str(tettaR[i]) + '\t\t\t\t\t\t' + str(resultEx3[i]) + '\n')
    f.write('Fstat = ' + str(Ff)+ '\n')

    resultEx4 = []

    #lw3:4 Вычисление гипотезы о незначимости гипотезы
    RRS = Get_RRS(y, X, tettaR)
    RRSH = Get_RRSH(y)
    F_valHip = Get_valHip(RRS, RRSH, N, len(tettaR))
    if(F_valHip < Ff):
        resultEx4.append('+')
    else:
        resultEx4.append('-')

    f.write('\n\nF\t\t\t\t\t\tQuantile of F-distribution\tHypothesis' + '\n')
    f.write(str(F_valHip) + '\t\t' + str(Ff) + '\t\t\t' + str(resultEx4[0]))
    
    #lw3:5 Вычисление доверительного интервала мат ожиадния и отклика
    uM = u_vec(x1, x2, tettaR)
    M_upper, M_lower, respond_upper, respond_lower, u_dov = Get_confidenceIntervalForMOandRespond(x1, x2, uM, sigma, sigmaR, tettaR, X) 
    create_plot(M_lower, u_dov, M_upper, np.sort(x1), t = 0) 
    create_plot(respond_lower, u_dov, respond_upper, np.sort(x1), t = 1) 
    #create_plot(M_lower, u_dov, M_upper, np.sort(x2), t = 0) 
    #create_plot(respond_lower, u_dov, respond_upper, np.sort(x2), t = 1)
    
    
    #SaveResultInTextFile(x1, x2, y, e, U)
    #SaveResultInCSVFile(x1, x2, y, e, U)
    plot.pause()
main()