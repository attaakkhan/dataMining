
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pywt
import seaborn as sns
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from scipy import fftpack

def f1(list1):
    list1 = list(list1)
    first=sum(list1[0:3])/3
    last = sum(list1[-3:]) / 3

    return last-first


def f2(list1):
    list1 = list(list1)
    X = fftpack.fft(list1)
    X=abs(X)
    X = list(X)
    X.sort(reverse=True)


    return X[1]*X[4]*X[5]*X[3]


def f3(list1):
    list1 = list(list1)
    mean=np.mean(list1)
    counts=0
    for i in list1:
        if(i>mean):
            counts = counts + 1
        else:
            counts = counts -1

    return counts


def f4(list1):
    list1 = list(list1)
    # print(list1[0])

    first = abs(list1[0])
    distance = 0.0
    for s in range(1, len(list1)):
        tmp = first - abs(float(list1[s]))
        first = abs(float(list1[s]))
        tmp = abs(tmp)
        distance = distance + tmp
    distance = distance
    return distance


def f6(list1):


    list1=list(list1)
    # cur=list1[0]
    # counts=0
    #
    # for i in list1[1:]:
    #     if i> cur:
    #         counts=counts+10
    #         cur=i

    counts=0
    if(list1[0]>list1[1] or list1[0]>list1[2]):
        counts=counts+30
    if (list1[1] < list1[2] or list1[1]< list1[3]):
        counts = counts + 30

    if(list1[1]<list1[7]):
        counts = counts + 30
    return counts



def f7(list):
    return np.std(list)


from scipy.stats import kurtosis, skew, entropy, variation, hmean, moment, median_absolute_deviation


def f8(list1):
    list1 = list(list1)
    X = fftpack.fft(list1)
    X = abs(X)
    X = list(X)
    X.sort(reverse=True)
    c=1;
    for i in X:
        c=c*i

    return  c

    return 0
    list1=list(list1)



    first = list1[0]
    counts=0
    j=0
    for i in range(1, 7):
        if (list1[i] -list1[i-1]  > 0):
            counts=counts+1
            j=i
        else:
            break

    if(4>j>0):
            if(list1[j] - list1[j-1]<0 and list1[0]<list1[-1]):
                          return 100

    return 0
    return dis
    cur=list1[0]
    counts=0

    for i in list1[1:]:
        if i> cur:
            counts=counts+10
            cur=i
    return counts

    list1 = list(list1)
    return kurtosis(list1)


def f9(list1):

   return skew(list1)


def f10(list1):
    list1 = list(list1)

    list1 = list(list1)

    first=list1[0]
    dis=0
    for i in range(1, 7):
        if(abs(first-list1[i])>5):
            dis=dis+100*i
    return dis
    return list1
    return entropy(list1)




def f11(list1):
    list1 = list(list1)
    return variation(list1)


def f12(list1):
    list1 = list(list1)
    return hmean(list1)


def f13(list1):
    list1 = list(list1)
    return moment(list1)


def f14(list1):
    list1 = list(list1)
    return median_absolute_deviation(list1)




def distance2(list1):
    list1 = list(list1)
    # print(list1[0])

    first = list1[0]
    distance = 0.0
    for s in range(1, len(list1)):
        tmp = first - float(list1[s])
        first = float(list1[s])
        tmp = abs(tmp)
        distance = distance + tmp
    distance = distance
    return distance


def distance3(list1):
    list1 = list(list1)
    # print(list1[0])

    first = list1[0]
    distance1 = 0.0
    distance2 = 0.0

    for s in range(1, len(list1)):
        tmp = float(list1[s]) - first
        first = float(list1[s])
        if tmp > 0:
            distance1 = distance1 + 10
        else:
            distance2 = distance2 - 10

    if (abs(distance2) > distance1):
        return distance2
    else:
        return distance1

    distance = distance
    return distance


def smin(list1):
    return np.min(list1)


def smax(list1):
    list1 = list(list1)
    # print(list1)
    return np.max(list1)

def smean(list1):
    list1 = list(list1)
    # print(list1)
    return np.mean(list1)






