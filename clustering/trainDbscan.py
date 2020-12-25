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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from scipy import fftpack

CO_RELLATION_METHOD = 'kendall'

SIMILARITY = .12
INTER_SIMILARITY = 0.85

MEAL_FILE = 'data/mealData{}.csv'
MEAL_AMOUNT_FILE = 'data/mealAmountData{}.csv'
NO_MEAL_FILE = 'data/Nomeal{}.csv'
MEAL_SAMPLES = 5;
NO_MEAL_SAMPLES = 5;
NORMALIZER_1 = 'bfill'
NORMALIZER_2 = 'pad'
TOP_PCA_FEATURES = 20


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


def apply_PCA1(df_in_1, ll):
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
        # print('***Loading file:{}'.format(file))
        df = pd.read_csv(file, names=cols, header=None)
        df1 = pd.read_csv(MEAL_AMOUNT_FILE.format(m + 1), names=['class'], header=None)
        # df.dropna(axis=0, inplace=True)

        # df=df.rolling(3).mean()
        # df.dropna(axis=0, inplace=True)
        # x = df.values  # returns a numpy array

        classes = [df1['class'].values[carb] for carb in range(0, df.shape[0])]
        # df.insert(0, "class", classes, True)
        targets = pd.cut(x=classes, bins=[-1, 0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4, 5], right=True)
        targets = list(targets)
        df.insert(0, "target", targets, True)
        # df.drop(['class'], inplace=True, axis=1)
        df.insert(0, "group", [m for trival in range(0, len(df))], True)
        meal_dfs.append(df)
    concat_df_list = []
    for subject in range(0, meal_samples):
        concat_df_list.append(meal_dfs[subject])
    final_df = pd.concat(concat_df_list, axis=0, sort=False)
   # print(final_df.shape)
    # print('asdsa')
    # final_df.insert(0, "time", [time for time in range(1, len(final_df) + 1)], True)
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
    cluster = []
    for i in range(0, 6):
        tmp = df[df['target'] == i]
        cluster.append(tmp)
    return cluster


def save_df_plot(final_df):
    for i in range(0, 260, 50):
        normal = final_df[final_df.time < i][final_df.columns]
        # print(normal.mean())
        normal[final_df.columns].plot(x="time", kind="line")
        plt.title('carbohydrates={}'.format(i))
        plt.savefig('data/plots/carbs{}.png'.format(i))


def attachCol(cols, word):
    cl = []
    for i in cols:
        cl.append('{}_{}'.format(i, word))
    return cl


def fft(list1):
    list1 = list(list1)
    X = fftpack.fft(list1)
    X = abs(X)
    X = list(X)
    X.sort(reverse=True)
    aa = X[0] + X[1] + [2]
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
    return sum(np.abs(list)) / 50


def displacement(list):
    return np.std(list)


from scipy.stats import kurtosis, skew, entropy, variation, hmean, moment, median_absolute_deviation


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
    return slop1(list1) + fft(list1)


def slop9(list1):
    list1 = list(list1)
    return slop5(list1) + slop2(list1)


def slop10(list1):
    list1 = list(list1)
    return distance(list1) + displacement(list1)


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
    men = 160
    list1 = list(list1)
    val = 0
    check = True
    check = list1[0];
    count = 0
    for i in range(1, len(list1)):

        if men - list1[i] < 0 and men - check > 0 or (men - list1[i] > 0 and men - check < 0):
            count = count + abs(((check - list1[i])))
            check = list1[i]

    return count


def slopadas(list1):
    list1 = list(list1)
    return list1[len(list1) - 1] - list1[3]


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

def get_row(r, i):
    # print (i)
    r = list(r)
    l = len(r)
    r1 = set(r)
    if (i == 0):
        return r[0:int(l / 2)]
    if (i == 1):
        return r[int(l / 2):l]
    if (i == 2):
        return r[int(l / 3):l]

    if (i == 3):
        return r[0:int(l / 3)]
    if (i == 4):
        return r[0:2 * int(l / 3)]
    if (i == 5):
        return r[0:l:2]
    if (i == 6):
        return r[0:l:3]
    if (i == 7):
        return r[0:l:5] + r[0:l:4]
    if (i == 8):
        return r[0:l:4] + r[0:l:7] + r[0:l:13]
    if (i == 9):
        return list(r1 - set((r[0:l:4] + r[0:l:7] + r[0:l:13])))
    if (i == 11):
        return list(r1 - set((r[0:l:5] + r[0:l:4])))
    if (i == 12):
        return list(r1 - set((r[0:l:3])))
    if (i == 13):
        return list(r1 - set((r[0:l:2])))
    if (i == 14):
        return list(r1 - set((r[0:2 * int(l / 3)])))
    if (i == 15):
        return list(r1 - set((r[0:int(l / 3)])))
    if (i == 16):
        return list(r1 - (set(r[0:l:4] + r[0:l:7] + r[0:l:13])))
    if (i == 17):
        return list(r1 - set((r[int(l / 3):l])))
    if (i == 18):
        return list(r1 - set((r[int(l / 2):l])))
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
        return r[4:6] + r[7:8] + r[13:19] + r[23:27]
    if (i == 26):
        return r[1:4] + r[9:12] + r[18:23] + r[28:30]
    if (i == 27):
        return r[4:6] + r[13:21] + r[26:27]
    if (i == 28):
        return r[1:2] + r[5:7] + r[11:15] + r[21:27]


def add_feature(df, feature_name, feature_function, con_list):
    df_rmax = df.copy()
    df_rmax.drop(['target'], inplace=True, axis=1)
    df_rmax.drop(['group'], inplace=True, axis=1)

    data = df_rmax.apply(lambda row: feature_function(row), axis=1)
    data = pd.DataFrame(data, columns=[feature_name])
    con_list.append(data)

    for ii in range(0, 29):
        # print (ii)
        data = df_rmax.apply(lambda row: feature_function(get_row(row, ii)), axis=1)
        data = pd.DataFrame(data, columns=[feature_name + '_' + str(ii)])
        con_list.append(data)
    for ii in range(0, 25, 8):
        data = df_rmax.apply(
            lambda row: feature_function(get_row(row, ii) + get_row(row, ii + 3) + get_row(row, ii + 1)), axis=1)
        data = pd.DataFrame(data, columns=[feature_name + '_' + str(ii) + '_' + str(ii + 1)])
        con_list.append(data)


def get_features1(df):


        con_list = []
        target = df['target']
        # add_feature(df,'rmax',smax,con_list)
        # add_feature(df, 'rmin', smin, con_list)
        # add_feature(df, 'rfft', fft, con_list)
        # #add_feature(df, 'rfft0', fft0, con_list)
        # add_feature(df, 'rfft1', fft1, con_list)
        # add_feature(df, 'rfft2', fft2, con_list)
        # add_feature(df, 'rvar', np.var, con_list)
      #  add_feature(df, 'rS0', slop0, con_list)
    #    add_feature(df, 'rS1', slop1, con_list)
     #   add_feature(df, 'rS2', slop2, con_list)
     #   add_feature(df, 'rS3', slop3, con_list)
     #    add_feature(df, 'rS4', slop4, con_list)
     #    add_feature(df, 'rS5', slop5, con_list)
     #    add_feature(df, 'rstd', np.std, con_list)
     #    add_feature(df, 'rmean', np.mean, con_list)
        add_feature(df, 'rdis2', distance2, con_list)
        add_feature(df, 'rdis', distance, con_list)
        add_feature(df, 'rdis3', distance3, con_list)
        #
        # add_feature(df, 'rS6', slop6, con_list)
        # add_feature(df, 'rS7', slop7, con_list)
        # add_feature(df, 'rS8', slop8, con_list)
        # add_feature(df, 'rS9', slop9, con_list)
        add_feature(df, 'rS10', slop10, con_list)







        dff = concat(con_list, 1)
        dff.insert(0, "target", target, True)
        #print(dff.head(5))
        return dff



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


def concat(dfs, axis):
    concat_df_list = []
    for target in range(0, len(dfs)):
        concat_df_list.append(dfs[target])
    final_df = pd.concat(dfs, axis=axis, sort=False)
    return final_df


def bisect(dff):
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(df)
    # df1 = pd.DataFrame(x_scaled)
    # df1 = normalize(df1)
    # df1 = (df - df.min()) / (df.max() - df.min())

    # for i in  range(0 ,5):
    df = dff.copy()
    df1 = dff.copy()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, max_iter=50000)
    kmeans.fit(df1)
    y_kmeans = kmeans.predict(df1)
    df.insert(0, "predict", y_kmeans, True)
    # print (df.shape[0])
    df0 = df[df['predict'] == 0].copy()
    df1 = df[df['predict'] == 1].copy()
    centers = kmeans.cluster_centers_
    df1.drop(['predict'], inplace=True, axis=1)
    df0.drop(['predict'], inplace=True, axis=1)
    # print (kmeans.inertia_)
    #  if i ==4:
    return (df0, centers[0], df1, centers[1])


def findMaxindex(listdf):
    max = 0
    index = 0
    for ii in range(0, len(listdf)):
        if listdf[ii].shape[0] > max:
            max = listdf[ii].shape[0]
            index = ii
    return index


def bisectKmean(df):
    df = df.copy()
    clusters = []
    centers = []

    for i in range(0, 5):
        if (len(clusters) == 0):
            res = bisect(df)
            clusters.append(res[0])
            centers.append(res[1])
            clusters.append(res[2])
            centers.append(res[3])
        else:
            maxIndex = findMaxindex(clusters)
            res = bisect(clusters[maxIndex])
            clusters.pop(maxIndex)
            centers.pop(maxIndex)
            clusters.append(res[0])
            centers.append(res[1])
            clusters.append(res[2])
            centers.append(res[3])
    return np.asarray(centers)


def bisectKmeanDB(df):
    df = df.copy()
    clusters = []
    centers = []

    for i in range(0, 5):
        if (len(clusters) == 0):
            res = bisect(df)
            clusters.append(res[0])
            centers.append(res[1])
            clusters.append(res[2])
            centers.append(res[3])
        else:
            maxIndex = findMaxindex(clusters)
            res = bisect(clusters[maxIndex])
            clusters.pop(maxIndex)
            centers.pop(maxIndex)
            clusters.append(res[0])
            centers.append(res[1])
            clusters.append(res[2])
            centers.append(res[3])
    return np.asarray(centers)


import math


def cal_distance(p1, p2):
    dis = []

    for i in range(0, len(p1)):
        dis.append(math.pow(p1[i] - p2[i], 2))
    return math.sqrt(np.sum(dis))


def cal_dis_pts(pts, center):
    dis = []
    # print (pts)
    for i in list(pts):
        # print(i)
        dis.append(cal_distance(i, center))
    return dis;


def cal_wiets(df):
    listw = [0, 0, 0, 0, 0, 0]
    totaldistance = np.sum(df.distance)
    df = df.sort_values('distance')
    # print(df)

    c0 = 0
    for index, row in df.iterrows():
        listw[int(row["target"])] = listw[int(row["target"])] + 1
        # if c0 > 40:
        #     break
        # c0 = c0 + 1
    return listw


def get_distance_of_N_from_each(df, centers):
    df = df.copy()
    dfs = []
    for i in range(0, 6):
        dfs.append(df[df['y_kmeans'] == i])
    # print(centers)
    clusters = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]]
    ff = []
    for i in range(0, 6):
        df = dfs[i][list(set(dfs[i].columns) - set(['y_kmeans', 'target']))].copy()
        distances = cal_dis_pts(np.asarray(df), centers[i])
        # print(len(distances))
        # print(df.shape)
        dfs[i].insert(0, "distance", distances, True)
        dfs[i] = dfs[i].sort_values('distance')
        aa = cal_wiets(dfs[i])

        ff.append(aa)

    # if aa[0]>15 or aa[1]>10 or aa[3]>10:

    check = False
    aa1 = np.asarray(ff)
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
    if np.max(aa1[0:5, 5]) > 2 and np.max(aa1[0:5, 0]) > 10 and np.max(aa1[0:5, 1]) > 10 and np.max(aa1[0:5, 3]) > 10:
        check = True
    check = True
    if check:
        for i in ff:
            print(i)
        # print('{}  {}'.format(cal_wiets(dfs[i]),np.argmax(cal_wiets(dfs[i]))))





def dbb(df11,target,cen):

            df1 = df11.copy()


            kmeans = KMeans(n_clusters=6, init=cen, n_init=1, max_iter=5000)
            kmeans.fit(df1)

            y_kmeans = kmeans.predict(df1)
            return y_kmeans
            centers = kmeans.cluster_centers_
            df1.insert(0, "y_kmeans", y_kmeans, True)
            df1.insert(0, "target", target, True)
            get_distance_of_N_from_each(df1, centers)
            df1.drop(['y_kmeans'], inplace=True, axis=1)
            df1.drop(['target'], inplace=True, axis=1)







cccccc=[[[ 0.04693578, -0.36739308,  0.28740402, -0.75548404,  0.04693578],
 [-0.152649  , -0.3170537  ,-0.62168331 , 0.44623302, -0.152649  ],
 [ 0.25132221 ,-0.15714113  ,0.81203776  ,0.11319692 , 0.25132221],
 [-0.2635947  ,-0.64399599 , 0.3319538   ,0.4423681  ,-0.2635947 ],
 [-0.39455816 , 0.35124017 ,-0.61875202  ,0.08983714 ,-0.39455816],
 [-0.3925008   ,0.76782866 , 0.05910067 ,-0.0194231 , -0.3925008 ]],
[[ 0.46036705 , 0.64203105,  0.03749951 , 0.31180103 , 0.46036705],
 [ 0.27537422 , 0.31527511,  0.70193611 , 0.37252447 , 0.27537422],
 [ 0.53475181 , 0.30402204, -0.15114074 , 0.50549506 , 0.53475181],
 [ 0.48769307 , 0.45428363,  0.10726303 , 0.53002214 , 0.48769307],
 [ 0.48140883 , 0.07175668,  0.35525971 , 0.60022102 , 0.48140883],
 [ 0.47989902 , 0.28921585 , 0.48449479 , 0.43745742 , 0.47989902]],
[[-0.58237275 ,-0.12166932 ,-0.0083766,  -0.50788633 ,-0.58237275],
 [-0.37573307 ,-0.44597332 ,-0.44475435 ,-0.47756418 ,-0.37573307],
 [-0.44859604 ,-0.54138402  ,0.12542371, -0.50239107 ,-0.44859604],
 [-0.47960614 ,-0.55448107 ,-0.17138217, -0.42507914 ,-0.47960614],
 [-0.41078677 ,-0.47545035 , 0.4633658 , -0.41304572 ,-0.41078677],
 [-0.18712238 ,-0.41702282 , 0.77223833 ,-0.30160947 ,-0.18712238]],
[[-0.03794077 , 0.1458711  ,-0.97370418 , 0.0093003  ,-0.03794077],
 [-0.18948566 , 0.02858183 ,-0.9347213  ,-0.19392476 ,-0.18948566],
 [ 0.37426784 , 0.38672508 ,-0.68941412 , 0.30480023 , 0.37426784],
 [ 0.39353539,  0.375528   ,-0.57692583 , 0.46272512 , 0.39353539],
 [ 0.15321322 , 0.39981789 ,-0.87322676 , 0.14692447,  0.15321322],
 [ 0.29236135 , 0.3067658,  -0.8236205   ,0.23789431 , 0.29236135]]]
def bisectdb(df):
    df.insert(0, "pos", [i for i in range(0,len(df))], True)
    poses=[]
    reses=[]
    dflist=[]
    for i in range(0,4):
     df1 = df[df['y_kmeans'] == i-1].copy()
     pos=list(df1['pos'])
     poses.append(pos)
     df1.drop(['pos'], inplace=True, axis=1)
    # print(pos)
     dby=df1['y_kmeans']
     df1.drop(['y_kmeans'], inplace=True, axis=1)
     target=df1['target']
     df1.drop(['target'], inplace=True, axis=1)
     #print(df1.shape)
     res=dbb(df1, list(target),np.asarray(cccccc[i]))

     if (i-1==-1):
         for i in range(0, len(res)):
             if res[i]==0:
                res[i]=2
             if res[i]==1:
                res[i]=1
             if res[i]==2:
                res[i]=2
             if res[i]==3:
                res[i]=3
             if res[i]==4:
                res[i]=0
             if res[i]==5:
                res[i]=2
     if (i-1 == 0):
                 for i in range(0, len(res)):
                     if res[i] == 0:
                         res[i] = 3
                     if res[i] == 1:
                         res[i] = 3
                     if res[i] == 2:
                         res[i] = 4
                     if res[i] == 3:
                         res[i] = 1
                     if res[i] == 4:
                         res[i] = 0
                     if res[i] == 5:
                         res[i] = 3
     if (i-1==1):
         for i in range(0, len(res)):
             if res[i]==0:
                res[i]=2
             if res[i]==1:
                res[i]=1
             if res[i]==2:
                res[i]=0
             if res[i]==3:
                res[i]=0
             if res[i]==4:
                res[i]=1
             if res[i]==5:
                res[i]=3
     if (i-1 == 2):
         for i in range(0, len(res)):
                     if res[i] == 0:
                          res[i] = 4
                     if res[i] == 1:
                          res[i] = 4
                     if res[i] == 2:
                          res[i] = 4
                     if res[i] == 3:
                          res[i] = 0
                          #print(res[i])
                     if res[i] == 4:
                          res[i] = 0

                     if res[i] == 5:
                          res[i] = 1





     reses.append(res)


     resDf = pd.DataFrame()
     resDf.insert(0, "res", res, True)
     resDf.insert(0, "pos", pos, True)
     dflist.append(resDf)



    resDf = pd.concat(dflist, axis=0, sort=False)
    resDf.sort_values(by=['pos'], inplace=True)
    target=list(df['target'])
    predict=list(resDf['res'])
    count=0
    for i in range(0 , len(target)):
        if target[i]==predict[i]:
            count=count+1


    #print(count)



from sklearn.cluster import KMeans, DBSCAN
import sys

df = get_single_df()
origanalDf=df.copy()
targets = df['target']
dff = get_features1(df)
#dff.to_csv('data/final_dffff4_NOTST.csv', index=False, header=True);


dff.dropna(axis=0, inplace=True)
target = dff['target']
#print(target)
targets = list(target)
dff.drop(['target'], inplace=True, axis=1)
list2 = ['rdis_17', 'rS10_18', 'rdis3_17', 'rdis_18','rdis_17']
dff1 = dff[list2]
cols = dff1.columns
scaler = StandardScaler()
x_scaled = scaler.fit_transform(dff1)
df1 = pd.DataFrame(x_scaled, columns=list2)

df1= normalize(df1)
df1 = pd.DataFrame(df1, columns=cols)
#print(df1.shape)
import pickle
dbsc = DBSCAN(eps=0.35 ,min_samples=5).fit(df1)
with open('traindbscan.pickle', 'wb') as f:
    pickle.dump(dbsc, f)
    print('exporting pickle')


