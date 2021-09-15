import numpy as np
import random
from scipy.linalg import *
from math import sqrt
from math import ceil
from matplotlib import pyplot as plt

lvl_x1=4
lvl_x2=5
N = 30
def u(x1,x2):
    return 1 + 1 * x1 + 0.002 * x1**2 + 2 * x2 + 0.003 * x2**2

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


x1,x2=getFactors()
U=[]

for i in range(N):
    U.append(round(u(x1[i], x2[i]),5))


print(U)
