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
from features import *

CO_RELLATION_METHOD = 'kendall'

SIMILARITY = .025
INTER_SIMILARITY = 0.4
FEATURES_FILE='features.csv'
MEAL_FILE = 'Meal_Prediction_DataMatrix.csv'
MEAL_AMOUNT_FILE = 'data/mealAmountData{}.csv'
NO_MEAL_FILE = 'NoMeal_Prediction_DataMatrix.csv'

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
    return topFeatures


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
    #print('PCA TOP Features:{}'.format(topFeatures))
    return topFeatures


def read_csv():
    df_m = pd.read_csv(MEAL_FILE,header=None)

    df_n = pd.read_csv(NO_MEAL_FILE, header=None)

    # df_m=handle_na(df_m)
    # df_n=handle_na(df_n)
    df_m=df_m.dropna(axis=0)
    df_n = df_n.dropna(axis=0)

    return [df_m,df_n]

def read_features():
    df = pd.read_csv(FEATURES_FILE,names=['label','f1','f2','f3','f4','f6','f7','f8','f9','f10','f11','f12','f13','f14','distance2','distance3','smin','smax','smean'], header=None)

    return df

def handle_na(df):
    df = df.dropna(thresh=2)
    df.fillna(method=NORMALIZER_1, inplace=True)
    df.fillna(method=NORMALIZER_2, inplace=True)
    return df


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



def add_feature(df, feature_name, feature_function, con_list):
    df_rmax = df.copy()

    data = df_rmax.apply(lambda row: feature_function(row), axis=1)
    data = pd.DataFrame(data, columns=[feature_name])
    con_list.append(data)


def get_features(df, test=None):

    con_list = []

    add_feature(df, 'f1', f1, con_list)
    add_feature(df, 'f2', f2, con_list)
    add_feature(df, 'f3', f3, con_list)
    add_feature(df, 'f4', f4, con_list)

    add_feature(df, 'f6', f6, con_list)
    add_feature(df, 'f7', f7, con_list)
    add_feature(df, 'f8', f8, con_list)
    add_feature(df, 'f9', f9, con_list)
    add_feature(df, 'f10', f10, con_list)
    add_feature(df, 'f11', f11, con_list)
    add_feature(df, 'f12', f12, con_list)
    add_feature(df, 'f13', f13, con_list)
    add_feature(df, 'f14', f14, con_list)
    add_feature(df, 'distance2', distance2, con_list)
    add_feature(df, 'distance3', distance3, con_list)
    add_feature(df, 'smin', smin, con_list)
    add_feature(df, 'smax', smin, con_list)
    add_feature(df, 'smean', smean, con_list)








    df = concat(con_list, 1)
    return df


def get_features1(df, l):

    con_list = []
    if 'f1' in l:
        add_feature(df, 'f1', f1, con_list)
    if 'f2' in l:
        add_feature(df, 'f2', f2, con_list)
    if 'f3' in l:
        add_feature(df, 'f3', f3, con_list)
    if 'f4' in l:
        add_feature(df, 'f4', f4, con_list)

    if 'f6' in l:
        add_feature(df, 'f6', f6, con_list)
    if 'f7' in l:
        add_feature(df, 'f7', f7, con_list)
    if 'f8' in l:
        add_feature(df, 'f8', f8, con_list)
    if 'f9' in l:
        add_feature(df, 'f9', f9, con_list)
    if 'f10' in l:
        add_feature(df, 'f10', f10, con_list)
    if 'f11' in l:
        add_feature(df, 'f11', f11, con_list)
    if 'f12' in l:
        add_feature(df, 'f12', f12, con_list)
    if 'f13' in l:
        add_feature(df, 'f13', f13, con_list)
    if 'f14' in l:
        add_feature(df, 'f14', f14, con_list)
    if 'distance2' in l:
        add_feature(df, 'distance2', distance2, con_list)
    if 'distance3' in l:
        add_feature(df, 'distance3', distance3, con_list)
    if 'smin' in l:
        add_feature(df, 'smin', smin, con_list)
    if 'smax' in l:
        add_feature(df, 'smax', smin, con_list)
    if 'smean' in l:
        add_feature(df, 'smean', smean, con_list)








    df = concat(con_list, 1)
    return df



def get_inter_independent_and_dependend_target_features(df_features, show=None):
    # df_features.insert(0, "target", labels, True)
    cor = df_features.corr(method=CO_RELLATION_METHOD)
    coll = list(df_features.columns)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('cor1.png')

    if (not show == None):
        plt.show()
    cor_target = abs(cor["label"])
    # print(cor_target)
    # print('asdsa')

    # print(i)
    relevant_features = cor_target[cor_target > SIMILARITY]
    features = list(relevant_features.index)


    print('Relevant features:{}'.format(features))
    dropList = []
    for i in range(1, len(features)):
        for j in range(1, len(features)):
            ind_cor = abs(df_features[[features[i], features[j]]].corr(method='kendall'))
            valu = ind_cor.iloc[0, 1]
            #   print (ind_cor)
            #  print(float(valu))
            if (float(valu) > INTER_SIMILARITY):
                #  print( 'KKKKK{}{}'.format(features[i], features[j]))
                if (not (features[i] == features[j])):
                    #     print(features[i], features[j])

                    if (abs(cor.iloc[coll.index(features[i]), coll.index('label')]) > abs(
                            cor.iloc[coll.index(features[j]), coll.index('label')])):
                        dropList.append((features[i], features[j]))
                    else:
                        dropList.append((features[j], features[i]))

   # print(dropList)

    for i in dropList:
        if i[0] in features and i[1] in features:
            features.remove(i[1])
    print('Selected features:{}'.format(features))

    dlist = list(set(df_features.columns) - set(features))
    df_features.drop(dlist, inplace=True, axis=1)
    #  df_features = df_features[features]

    return df_features



def concat(dfs, axis):
    concat_df_list = []
    for target in range(0, len(dfs)):
        concat_df_list.append(dfs[target])
    final_df = pd.concat(dfs, axis=axis, sort=False)
    return final_df





def get_and_save_features():
    res = read_csv()
    df_m = res[0]
    df_n = res[1]

    df_m = get_features(df_m)
    df_n = get_features(df_n)
    df_n.insert(0, "label", [1] * df_n.shape[0], True)
    df_m.insert(0, "label", [0] * df_m.shape[0], True)

    features = concat([df_n, df_m], 0)
    features.to_csv(FEATURES_FILE, index=False, header=False)


def get_features_from_list(l):
    res = read_csv()
    df_m = res[0]
    df_n = res[1]

    df_m = get_features1(df_m,l)
    df_n = get_features1(df_n,l)
    df_n.insert(0, "label", [1] * df_n.shape[0], True)
    df_m.insert(0, "label", [0] * df_m.shape[0], True)

    features = concat([df_n, df_m], 0)
    return features

import pickle

def store_features_list(l):
    with open('list_features.txt', 'w') as filehandle:
        for listitem in l:
            filehandle.write('%s\n' % listitem)

import json

def load_features_list():
    places = []
    with open('list_features.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            places.append(currentPlace)
        return places


import time


def extract():
    start_time = time.time()
    get_and_save_features()

    features=read_features()
    print("Feature extracted:"+str(list(features.columns)))
    selected_features=get_inter_independent_and_dependend_target_features(features)
    print("Features Selected after corellation"+str(list(selected_features.columns)))
    selected_features=apply_PCA1(selected_features,10)
    print("Featurs After PCA, In Decender order:"+str(list(selected_features)))
    store_features_list(selected_features)
    print("******Execution time for extracting features: %s seconds ---" % (time.time() - start_time))
    # features_list=load_features_list()
    # print(get_features_from_list(features_list[0:5]))
    return







