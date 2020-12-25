import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pywt
import seaborn as sns
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from scipy import fftpack
CO_RELLATION_METHOD = 'kendall'

SIMILARITY =.12
INTER_SIMILARITY=0.85









MEAL_FILE = 'data/mealData{}.csv'
MEAL_AMOUNT_FILE = 'data/mealAmountData{}.csv'
NO_MEAL_FILE = 'data/Nomeal{}.csv'
MEAL_SAMPLES = 5;
NO_MEAL_SAMPLES = 5;
NORMALIZER_1 = 'bfill'
NORMALIZER_2 = 'pad'
TOP_PCA_FEATURES=20

def apply_PCA(df_in_1):
    df_in = df_in_1.copy()
    comCol = df_in.columns
    pca = PCA(n_components=2)
    pca.fit(df_in)

    columns = ['pca_%i' % i for i in range(2)]
    df_pca = DataFrame(pca.transform(df_in), columns=columns, index=df_in.index)
    # pca_score = pca.explained_variance_ratio
    comp = pca.components_
    comp1 = comp[0]
    comp1 = np.abs(comp1)
    comp1 = np.abs(comp1)
    top = comp1.argsort()[-TOP_PCA_FEATURES:][::-1]
    # print (top)
    topFeatures = []
    for tt in top:
        topFeatures.append(comCol[tt])
    print('PCA TOP Features:{}'.format(topFeatures))
    return df_in_1[topFeatures]

def apply_PCA1(df_in_1,ll):
    df_in = df_in_1.copy()
    comCol = df_in.columns
    pca = PCA(n_components=2)
    pca.fit(df_in)

    columns = ['pca_%i' % i for i in range(2)]
    df_pca = DataFrame(pca.transform(df_in), columns=columns, index=df_in.index)
    # pca_score = pca.explained_variance_ratio
    comp = pca.components_
    comp1 = comp[0]
    comp1 = np.abs(comp1)
    comp1 = np.abs(comp1)
    top = comp1.argsort()[-ll:][::-1]
    # print (top)
    topFeatures = []
    for tt in top:
        topFeatures.append(comCol[tt])
    print('PCA TOP Features:{}'.format(topFeatures))
    return df_in_1[topFeatures]

def get_single_df():

    meal_samples = MEAL_SAMPLES
    non_meal_samples = NO_MEAL_SAMPLES
    meal_dfs = []
    no_meal_dfs = []
    no_of_attribute = 31;
    cols = ["series_{}".format(i) for i in range(1, no_of_attribute + 1)]
    y = [True, False] * 5
    y = np.array(y)
    y = pd.Series(y, name='target')

    for m in range(0, meal_samples):
        file = MEAL_FILE.format(m + 1)
        #print('***Loading file:{}'.format(file))
        df = pd.read_csv(file, names=cols, header=None)
        df1 = pd.read_csv(MEAL_AMOUNT_FILE.format(m + 1), names=['class'], header=None)
        #df.dropna(axis=0, inplace=True)


       # df=df.rolling(3).mean()
        #df.dropna(axis=0, inplace=True)
        #x = df.values  # returns a numpy array


        classes=[ df1['class'].values[carb] for carb in range(0, df.shape[0])]
       # df.insert(0, "class", classes, True)
        targets = pd.cut(x=classes, bins=[-1, 0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4, 5] ,right=True)
        targets = list(targets)
        df.insert(0, "target", targets, True)
        #df.drop(['class'], inplace=True, axis=1)
        df.insert(0, "group", [m for trival in range(0, len(df))], True)
        meal_dfs.append(df)
    concat_df_list = []
    for subject in range(0, meal_samples):
        concat_df_list.append(meal_dfs[subject])
    final_df = pd.concat(concat_df_list, axis=0, sort=False)
  #  print(final_df.shape)
    #print('asdsa')
    #final_df.insert(0, "time", [time for time in range(1, len(final_df) + 1)], True)
    final_df = final_df.dropna(thresh=15)
    final_df.fillna(method=NORMALIZER_1, inplace=True)
    final_df.fillna(method=NORMALIZER_2, inplace=True)
    return final_df


def myremovena(final_df):
    final_df = final_df.dropna(thresh=4)
    final_df.fillna(method=NORMALIZER_1, inplace=True)
    final_df.fillna(method=NORMALIZER_2, inplace=True)
    return final_df
def get_clustuers_dfs(df):
    cluster=[]
    for i in range(0,6):
        tmp= df[df['target'] == i]
        cluster.append(tmp)
    return cluster


def save_df_plot(final_df):
    for i in range(0, 260,50):
        normal = final_df[final_df.time< i][final_df.columns]
       # print(normal.mean())
        normal[final_df.columns].plot(x="time", kind="line")
        plt.title('carbohydrates={}'.format(i))
        plt.savefig('data/plots/carbs{}.png'.format(i))

def attachCol(cols,word):
    cl=[]
    for i in cols:
        cl.append('{}_{}'.format(i,word))
    return cl


def fft(list1):

        list1=list(list1)
        X = fftpack.fft(list1)
        X = abs(X)
        X = list(X)
        X.sort(reverse=True)
        aa=X[0]+X[1]+[2]
        return aa[0]


def fft0(list1):
    list1 = list(list1)
    X = fftpack.fft(list1)
    X = abs(X)
    X = list(X)
    X.sort(reverse=True)

    return X[0]


def fft1(list1):
    list1 = list(list1)
    X = fftpack.fft(list1)
    X = abs(X)
    X = list(X)
    X.sort(reverse=True)

    return X[1]


def fft2(list1):
    list1 = list(list1)
    X = fftpack.fft(list1)
    X = abs(X)
    X = list(X)
    X.sort(reverse=True)

    return X[2]



def distance(list):

       return  sum(np.abs(list)) / 50

def displacement(list):
   return np.std(list)
from scipy.stats import kurtosis, skew,entropy, variation,hmean, moment, median_absolute_deviation

def slop0(list1):
   list1 = list(list1)
   return kurtosis(list1)
def slop1(list1):
   list1 = list(list1)
   return skew(list1)
def slop2(list1):
   list1 = list(list1)
   return entropy(list1)
def slop8(list1):
   list1 = list(list1)
   return slop1(list1)+fft(list1)

def slop9(list1):
   list1 = list(list1)
   return slop5(list1)+slop2(list1)
def slop10(list1):
   list1 = list(list1)
   return distance(list1)+displacement(list1)



def slop3(list1):
   list1 = list(list1)
   return variation(list1)
def slop4(list1):
   list1 = list(list1)
   return hmean(list1)


def slop5(list1):
   list1 = list(list1)
   return moment(list1)

def slop6(list1):
   list1 = list(list1)
   return median_absolute_deviation(list1)

def slop7(list1):
   men=160
   list1 = list(list1)
   val=0
   check=True
   check=list1[0];
   count=0
   for i in range(1,len(list1)):

       if men-list1[i]<0 and men-check>0 or (men-list1[i]>0 and men-check<0):
           count =count +abs(((check-list1[i])))
           check=list1[i]


   return count
def slopadas(list1):
   list1 = list(list1)
   return list1[len(list1)-1]-list1[3]



def distance2(list1):
        list1=list(list1)
        #print(list1[0])


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
        tmp = float(list1[s])-first
        first = float(list1[s])
        if tmp>0:
            distance1 = distance1 + 10
        else:
            distance2=distance2-10

    if(abs(distance2)>distance1):
                return  distance2
    else:
                return distance1


    distance = distance
    return distance

def smin(list1):
    return np.min(list1)
def smax(list1):
    list1=list(list1)
    #print(list1)
    return np.max(list1)
# def get_features(df):
#     dfs=[]
#     target=df['target']
#     for i in range(0,5):
#         df1=df[df['group']==i].diff()
#         dfs.append(df1)
#     dfS=concat(dfs,0)
#     dfS.drop(['target'], inplace=True, axis=1)
#     dfS.drop(['group'], inplace=True, axis=1)
#     dfS.columns=attachCol(dfS.columns,'diff')
#     dfs=[]
#     for i in range(0, 5):
#         df1 = df[df['group'] == i]
#         df1=df1-df1.mean()
#         dfs.append(df1)
#     dfM=concat(dfs,0)
#     dfM.drop(['target'], inplace=True, axis=1)
#     dfM.drop(['group'], inplace=True, axis=1)
#     dfM.columns = attachCol(dfM.columns, 'mean')
#
#     dfs = []
#     target = df['target']
#     for i in range(0, 5):
#         df1 = df[df['group'] == i].copy()
#         df1=df1.rolling(5).apply(distance2)
#         dfs.append(df1)
#     dfD = concat(dfs, 0)
#     dfD.drop(['target'], inplace=True, axis=1)
#     dfD.drop(['group'], inplace=True, axis=1)
#     dfD.columns = attachCol(dfD.columns, 'distance2')
#
#     dfs = []
#     target = df['target']
#     for i in range(0, 5):
#         df1 = df[df['group'] == i].copy()
#         df1 = df1.rolling(8).apply(displacement)
#         dfs.append(df1)
#     dfd = concat(dfs, 0)
#     dfd.drop(['target'], inplace=True, axis=1)
#     dfd.drop(['group'], inplace=True, axis=1)
#     dfd.columns = attachCol(dfd.columns, 'displacement')
#
#     dfs = []
#     target = df['target']
#     for i in range(0, 5):
#         df1 = df[df['group'] == i].copy()
#         df1 = df1.rolling(8).apply(distance)
#         dfs.append(df1)
#     dfd1 = concat(dfs, 0)
#     dfd1.drop(['target'], inplace=True, axis=1)
#     dfd1.drop(['group'], inplace=True, axis=1)
#     dfd1.columns = attachCol(dfd1.columns, 'distance1')
#
#     dfs = []
#     target = df['target']
#     for i in range(0, 5):
#         df1 = df[df['group'] == i].copy()
#         df1 = df1.rolling(4).apply(fft)
#         dfs.append(df1)
#     dfft = concat(dfs, 0)
#     dfft.drop(['target'], inplace=True, axis=1)
#     dfft.drop(['group'], inplace=True, axis=1)
#     dfft.columns = attachCol(dfft.columns, 'fft')
#
#
#
#
#
#
#
#
#     df.drop(['target'], inplace=True, axis=1)
#     df.drop(['group'], inplace=True, axis=1)
#
#     dff=concat([dfS,dfM,dfD,dfft,dfd,dfd1],1)
#     dff.insert(0, "target", target, True)
#     return dff

def get_row(r,i):
   # print (i)
    r=list(r)
    l=len(r)
    r1=set(r)
    if(i==0):
        return r[0:int(l/2)]
    if (i == 1):
        return r[int(l / 2):l]
    if (i == 2):
        return r[int(l/3):l]

    if (i == 3):
        return r[0:int(l/3)]
    if (i == 4):
        return r[0:2 * int(l / 3)]
    if (i == 5):
        return  r[0:l:2]
    if (i == 6):
        return r[0:l:3]
    if (i == 7):
        return r[0:l:5]+ r[0:l:4]
    if (i == 8):
        return r[0:l:4]+r[0:l:7]+r[0:l:13]
    if (i == 9):
        return list(r1- set((r[0:l:4]+r[0:l:7]+r[0:l:13])))
    if (i == 11):
        return list(r1 - set((r[0:l:5]+ r[0:l:4])))
    if (i == 12):
        return list(r1 - set((  r[0:l:3])))
    if (i == 13):
        return list(r1 - set((   r[0:l:2])))
    if (i == 14):
        return  list(r1 - set((r[0:2 * int(l / 3)])))
    if (i == 15):
        return list(r1 - set(( r[0:int(l/3)])))
    if (i == 16):
        return list(r1 - (set(r[0:l:4] + r[0:l:7] + r[0:l:13])))
    if (i == 17):
        return list(r1 - set(( r[int(l/3):l])))
    if (i == 18):
        return list(r1 - set(( r[int(l / 2):l])))
    if (i == 10):
        return list(r1 - set((r[int(l / 2):l])))


    if (i == 19):
        return r[0:5]
    if (i == 20):
        return r[5:10]
    if (i == 21):
        return r[10:15]
    if (i == 22):
        return r[15:20]
    if (i == 23):
        return r[20:25]
    if (i == 24):
        return r[25:30]
    if (i == 25):
        return r[4:6]+r[7:8]+r[13:19]+r[23:27]
    if (i == 26):
        return r[1:4]+r[9:12]+r[18:23]+r[28:30]
    if (i == 27):
        return r[4:6]+r[13:21]+r[26:27]
    if (i == 28):
        return r[1:2]+r[5:7]+r[11:15]+r[21:27]













def add_feature(df,feature_name,feature_function, con_list):
    df_rmax=df.copy()
    df_rmax.drop(['target'], inplace=True, axis=1)
    df_rmax.drop(['group'], inplace=True, axis=1)

    data=df_rmax.apply(lambda row: feature_function(row), axis = 1)
    data = pd.DataFrame(data, columns=[feature_name])
    con_list.append(data)

    for ii in range(0,29):
        #print (ii)
        data=df_rmax.apply(lambda row: feature_function(get_row(row,ii)), axis = 1)
        data = pd.DataFrame(data, columns=[feature_name+'_'+str(ii)])
        con_list.append(data)
    # for ii in range(0,25,8):
    #
    #     data=df_rmax.apply(lambda row: feature_function(get_row(row,ii)+get_row(row,ii+3)+get_row(row,ii+1)), axis = 1)
    #     data = pd.DataFrame(data, columns=[feature_name+'_'+str(ii)+'_'+str(ii+1)])
    #     con_list.append(data)

def get_features1(df,test=None):
    # dfs=[]
    # for i in range(0, 5):
    #     df1 = df[ df['group'] == i]
    #     df1=df1-df1.mean()
    #     dfs.append(df1)
    # dfS0=concat(dfs,0)
    # dfS0.drop(['target'], inplace=True, axis=1)
    # dfS0.drop(['group'], inplace=True, axis=1)
    # dfS0.columns = attachCol(dfS0.columns, 'S2')

    con_list = []
    if(test==None):
     target = df['target']
#     add_feature(df,'rmax',smax,con_list)
#     add_feature(df, 'rmin', smin, con_list)
#     add_feature(df, 'rfft', fft, con_list)
#     #add_feature(df, 'rfft0', fft0, con_list)
#     add_feature(df, 'rfft1', fft1, con_list)
#     add_feature(df, 'rfft2', fft2, con_list)
#     add_feature(df, 'rvar', np.var, con_list)
#   #  add_feature(df, 'rS0', slop0, con_list)
# #    add_feature(df, 'rS1', slop1, con_list)
#  #   add_feature(df, 'rS2', slop2, con_list)
#  #   add_feature(df, 'rS3', slop3, con_list)
#     add_feature(df, 'rS4', slop4, con_list)
#     add_feature(df, 'rS5', slop5, con_list)
#     add_feature(df, 'rstd', np.std, con_list)
#     add_feature(df, 'rmean', np.mean, con_list)
    add_feature(df, 'rdis2', distance2, con_list)
    add_feature(df, 'rdis', distance, con_list)
    add_feature(df, 'rdis3', distance3, con_list)

    # add_feature(df, 'rS6', slop6, con_list)
    # add_feature(df, 'rS7', slop7, con_list)
    # add_feature(df, 'rS8', slop8, con_list)
    # add_feature(df, 'rS9', slop9, con_list)
    add_feature(df, 'rS10', slop10, con_list)
#
#
#
#
#
#
#
    dff = concat(con_list, 1)
    if (test == None):
     dff.insert(0, "target", target, True)
    #print(dff.head(5))
   # print('jajja')
    return dff














    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop1)
        dfs.append(df1)
    dfS1 = concat(dfs, 0)
    dfS1.drop(['target'], inplace=True, axis=1)
    dfS1.drop(['group'], inplace=True, axis=1)
    dfS1.columns = attachCol(dfS1.columns, 'S1')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop2)
        dfs.append(df1)
    dfS2 = concat(dfs, 0)
    dfS2.drop(['target'], inplace=True, axis=1)
    dfS2.drop(['group'], inplace=True, axis=1)
    dfS2.columns = attachCol(dfS2.columns, 'entropy')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop6)
        dfs.append(df1)
    dfS6 = concat(dfs, 0)
    dfS6.drop(['target'], inplace=True, axis=1)
    dfS6.drop(['group'], inplace=True, axis=1)
    dfS6.columns = attachCol(dfS2.columns, 'S6')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop5)
        dfs.append(df1)
    dfS5 = concat(dfs, 0)
    dfS5.drop(['target'], inplace=True, axis=1)
    dfS5.drop(['group'], inplace=True, axis=1)
    dfS5.columns = attachCol(dfS5.columns, 'S5')





    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop4)
        dfs.append(df1)
    dfS4=concat(dfs,0)
    dfS4.drop(['target'], inplace=True, axis=1)
    dfS4.drop(['group'], inplace=True, axis=1)
    dfS4.columns = attachCol(dfS4.columns, 'S4')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop3)
        dfs.append(df1)
    dfS3=concat(dfs,0)
    dfS3.drop(['target'], inplace=True, axis=1)
    dfS3.drop(['group'], inplace=True, axis=1)
    dfS3.columns = attachCol(dfS3.columns, 'S3')


    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop0)
        dfs.append(df1)
    dfS0=concat(dfs,0)
    dfS0.drop(['target'], inplace=True, axis=1)
    dfS0.drop(['group'], inplace=True, axis=1)
    dfS0.columns = attachCol(dfS0.columns, 'S0')


    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(slop7)
        dfs.append(df1)
    dfS7=concat(dfs,0)
    dfS7.drop(['target'], inplace=True, axis=1)
    dfS7.drop(['group'], inplace=True, axis=1)
    dfS7.columns = attachCol(dfS7.columns, 'S7')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1=df1.rolling(3).apply(distance2)
        dfs.append(df1)
    dfD = concat(dfs, 0)
    dfD.drop(['target'], inplace=True, axis=1)
    dfD.drop(['group'], inplace=True, axis=1)
    dfD.columns = attachCol(dfD.columns, 'distance2')
    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(distance3)
        dfs.append(df1)
    dfD3 = concat(dfs, 0)
    dfD3.drop(['target'], inplace=True, axis=1)
    dfD3.drop(['group'], inplace=True, axis=1)
    dfD3.columns = attachCol(dfD3.columns, 'distance3')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(fft)
        dfs.append(df1)
    dfft = concat(dfs, 0)
    dfft.drop(['target'], inplace=True, axis=1)
    dfft.drop(['group'], inplace=True, axis=1)
    dfft.columns = attachCol(dfft.columns, 'fft')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(displacement)
        dfs.append(df1)
    dfd = concat(dfs, 0)
    dfd.drop(['target'], inplace=True, axis=1)
    dfd.drop(['group'], inplace=True, axis=1)
    dfd.columns = attachCol(dfd.columns, 'displacement')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(smin)
        dfs.append(df1)
    dfdm = concat(dfs, 0)
    dfdm.drop(['target'], inplace=True, axis=1)
    dfdm.drop(['group'], inplace=True, axis=1)
    dfdm.columns = attachCol(dfdm.columns, 'min')

    dfs = []
    target = df['target']
    for i in range(0, 5):
        df1 = df[df['group'] == i].copy()
        df1 = df1.rolling(3).apply(smax)
        dfs.append(df1)
    dfdmm = concat(dfs, 0)
    dfdmm.drop(['target'], inplace=True, axis=1)
    dfdmm.drop(['group'], inplace=True, axis=1)
    dfdmm.columns = attachCol(dfdmm.columns, 'max')


    df.drop(['target'], inplace=True, axis=1)
    df.drop(['group'], inplace=True, axis=1)

    dff=concat([dfS0,dfS1,dfS2,dfS3,dfS4,dfS5,dfS6,dfS7,dfD,dfD3,df,dfft,dfd,dfdm,dfdmm],1)
    dff.insert(0, "target", target, True)
    print(dff.head(5))
    return dff






def get_inter_independent_and_dependend_target_features(df_features, labels, show=None):
   # df_features.insert(0, "target", labels, True)
    cor = df_features.corr(method=CO_RELLATION_METHOD)
    coll = list(df_features.columns)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('data/plots/cor1.png')



    if (not show == None):
        plt.show()
    cor_target = abs(cor["target"])
    #print(cor_target)
    #print('asdsa')

   # print(i)
    relevant_features = cor_target[cor_target > SIMILARITY]
    features = list(relevant_features.index)

    features.remove('target')
    print('Relevant features:{}'.format(features))
    dropList = []
    for i in range(1, len(features)):
        for j in range(1, len(features)):
            ind_cor = abs(df_features[[features[i], features[j]]].corr(method='kendall'))
            valu=ind_cor.iloc[0, 1]
            #   print (ind_cor)
          #  print(float(valu))
            if (float(valu)>INTER_SIMILARITY):
              #  print( 'KKKKK{}{}'.format(features[i], features[j]))
                if (not (features[i] == features[j])):
               #     print(features[i], features[j])

                    if (abs(cor.iloc[coll.index(features[i]), coll.index('target')]) > abs(
                            cor.iloc[coll.index(features[j]), coll.index('target')])):
                        dropList.append((features[i], features[j]))
                    else:
                        dropList.append((features[j], features[i]))

    for i in dropList:
        if i[0] in features and i[1] in features:
            features.remove(i[1])
    print('Selected features:{}'.format(features))

    dlist=list(set(df_features.columns)-set(features))
    df_features.drop(dlist, inplace=True, axis=1)
  #  df_features = df_features[features]

    return df_features;








# def get_features(clusturs):
#     for i in clusturs:
#         i=i-i.mean()
#     return clusturs;

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num



def concat(dfs,axis):
    concat_df_list = []
    for target in range(0,len(dfs)):
        concat_df_list.append(dfs[target])
    final_df = pd.concat(dfs, axis=axis, sort=False)
    return final_df




def bisect(dff):
    #scaler = StandardScaler()
    #x_scaled = scaler.fit_transform(df)
    #df1 = pd.DataFrame(x_scaled)
    # df1 = normalize(df1)
    # df1 = (df - df.min()) / (df.max() - df.min())

 # for i in  range(0 ,5):
    df=dff.copy()
    df1 = dff.copy()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, max_iter=50000)
    kmeans.fit(df1)
    y_kmeans = kmeans.predict(df1)
    df.insert(0, "predict", y_kmeans, True)
   # print (df.shape[0])
    df0=df[df['predict']==0].copy()
    df1 = df[df['predict'] == 1].copy()
    centers = kmeans.cluster_centers_
    df1.drop(['predict'], inplace=True, axis=1)
    df0.drop(['predict'], inplace=True, axis=1)
    #print (kmeans.inertia_)
  #  if i ==4:
    return (df0,centers[0], df1,centers[1])

def findMaxindex(listdf):
    max=0
    index=0
    for ii in range(0 ,len(listdf)):
        if listdf[ii].shape[0]>max:
            max= listdf[ii].shape[0]
            index=ii
    return index

def bisectKmean(df):
    df=df.copy()
    clusters=[]
    centers=[]

    for i in range(0,5):
        if(len(clusters)==0):
            res=bisect(df)
            clusters.append(res[0])
            centers.append(res[1])
            clusters.append(res[2])
            centers.append(res[3])
        else:
            maxIndex=findMaxindex(clusters)
            res=bisect(clusters[maxIndex])
            clusters.pop(maxIndex)
            centers.pop(maxIndex)
            clusters.append(res[0])
            centers.append(res[1])
            clusters.append(res[2])
            centers.append(res[3])
    return np.asarray(centers)


import math
def cal_distance(p1,p2):
    dis=[]

    for i in range(0,len(p1)):
        dis.append(math.pow(p1[i]-p2[i],2))
    return math.sqrt(np.sum(dis))



def cal_dis_pts(pts,center):
    dis=[]
    #print (pts)
    for i in list(pts):
        #print(i)
        dis.append(cal_distance(i,center))
    return dis;

def cal_wiets(df):
    listw=[0,0,0,0,0,0]
    totaldistance=np.sum(df.distance)
    df=df.sort_values('distance')
    #print(df)

    c0=0
    for index, row in df.iterrows():
       listw[int(row["target"])]=listw[int(row["target"])]+1
       if c0>40:
            break
       c0=c0+1
    return listw
def get_distance_of_N_from_each(df,centers):
    df=df.copy()
    dfs=[]
    for i in range(0,6):
        dfs.append(df[df['y_kmeans']==i])
    #print(centers)
    clusters=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    ff=[]
    for i in range(0,6):
         df=dfs[i][list(set(dfs[i].columns)-set(['y_kmeans','target']))].copy()
         distances=cal_dis_pts(np.asarray(df),centers[i])
         #print(len(distances))
         #print(df.shape)
         dfs[i].insert(0, "distance", distances, True)
         dfs[i]=dfs[i].sort_values('distance')
         aa=cal_wiets(dfs[i])

         ff.append(aa)

        # if aa[0]>15 or aa[1]>10 or aa[3]>10:


    check=False
    aa1=np.asarray(ff)
    #
    # for ii in range(0 ,6):
    #
    #  if np.max(aa1[0:5,ii]):
    #
    #
    # if ff[ii][0]>18:
    #         check=True
    #         break
    # if ff[ii][1] > 15:
    #         check = True
    #         break
    # if ff[ii][3] > 15:
    #
    #         check = True
    #         break
    if np.max(aa1[0:5, 5])>2 and np.max(aa1[0:5, 0])>18 and np.max(aa1[0:5, 1])>8 and np.max(aa1[0:5, 3])>10:
        check=True
    
    if check:
        for i in ff:
            print (i)
        #print('{}  {}'.format(cal_wiets(dfs[i]),np.argmax(cal_wiets(dfs[i]))))




def train():
    import sys

    df = get_single_df()
    targets = df['target']
    dff = get_features1(df)

    # print(dff.columns)
    dff.dropna(axis=0, inplace=True)

    targets = dff['target']

    targets = list(targets)

    import random
    list2 = ['rS7_23', 'rdis_18', 'rdis3_17', 'rmean_17', 'rdis_17', 'rmin_19', 'rS9_18']
    # list2=['rdis3_4', 'rmin_19', 'rS4_17', 'rmax_17', 'rS10_18' , 'rdis_18', 'rdis_17', 'rstd_27', 'rS7_22', 'rdis3_17', 'rstd_24']
    # list2=['series_3_fft', 'series_4_max', 'series_4', 'series_1', 'series_2_S4', 'series_5_S4', 'series_1_S4', 'series_3_min', 'series_8_S4', 'series_6_min', 'series_1_distance3','series_31_min'] #'series_8_S7', 'series_11_S7', 'series_31_min', 'series_31_S4', 'series_1_distance3']
    list2 = ['rdis_18', 'rdis3_17', 'rmean_17', 'rdis_17', 'rmin_19', 'rS9_18']
    list2 = ['rdis_17', 'rS10_18', 'rdis3_17', 'rdis_18', 'rdis_17']

    listttt = list2
    for kk in range(0, 1):

        dff1 = dff[list2]

        cols = dff1.columns
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(dff1)
        df11 = pd.DataFrame(x_scaled, columns=cols)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(dff)
        df2 = pd.DataFrame(x_scaled)

        # df1 = (df - df.min()) / (df.max() - df.min())
        from sklearn.cluster import KMeans
        for ll in range(0, 1):
            df1 = df11.copy()
            #  centers = bisectKmean(df1)

            for i in range(99, 100):
                ceters = np.array([[-1.21606013, -1.22412298, -0.0557472, -1.24532652, -1.21606013],
                                   [0.85902053, 0.67521048, 0.46555977, 0.83603056, 0.85902053],
                                   [2.15549909, 2.03729549, -0.28774145, 2.14119884, 2.15549909],
                                   [0.15824922, 0.99763724, -3.32109638, 0.1963196, 0.15824922],
                                   [0.04012781, 0.07547363, 0.1779027, 0.04955957, 0.04012781],
                                   [-0.58549039, -0.56632798, 0.21451573, -0.54830571, -0.58549039]])
                kmeans = KMeans(n_clusters=6, init=ceters, n_init=1, max_iter=100000)
                kmeans.fit(df1)
                # print(df1)

                y_kmeans = kmeans.predict(df1)
                centers = kmeans.cluster_centers_
                # print(centers)

                # df1.colums = cols
                df1.insert(0, "y_kmeans", y_kmeans, True)
                df1.insert(0, "target", targets, True)
                # print(df1[df1['y_kmeans']==2])
                import pickle
                if (i == 99):
                    with open('trainkmean.pickle', 'wb') as f:
                        pickle.dump(kmeans, f)
                       # print('exporting pickle')
                   # print(centers)
                    #get_distance_of_N_from_each(df1, centers)
                df1.drop(['y_kmeans'], inplace=True, axis=1)
                df1.drop(['target'], inplace=True, axis=1)
                return kmeans






train()
