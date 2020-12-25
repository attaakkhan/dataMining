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
import time
import matplotlib.pyplot as plt
import numpy as np
from features import *

import seaborn as sns
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from math import sqrt
from sklearn.linear_model import LogisticRegression
import math
from sklearn.model_selection import KFold  # import KFold
import pickle
import os
import csv
CO_RELLATION_METHOD = 'kendall'

SIMILARITY = .01
INTER_SIMILARITY = 0.85
FEATURES_FILE='features.csv'
MEAL_FILE = 'Meal_Prediction_DataMatrix.csv'

NO_MEAL_FILE = 'NoMeal_Prediction_DataMatrix.csv'

NORMALIZER_1 = 'bfill'
NORMALIZER_2 = 'pad'


def read_csv():
    df_m = pd.read_csv(MEAL_FILE,header=None)

    df_n = pd.read_csv(NO_MEAL_FILE, header=None)

    df_m=df_m.dropna(axis=0)
    df_n = df_n.dropna(axis=0)

    # df_m=handle_na(df_m)
    # df_n=handle_na(df_n)
    return [df_m,df_n]

def read_features():
    df = pd.read_csv(FEATURES_FILE,names=['label','f1','f2','f3','f4','f6','f7','f8','f9','f10','f11','f12','f13','f14','distance2','distance3','smin','smax','smean'], header=None)

    return df

def handle_na(df):
    df = df.dropna(thresh=2)
    df.fillna(method=NORMALIZER_1, inplace=True)
    df.fillna(method=NORMALIZER_2, inplace=True)
    return df



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


    #print('Relevant features:{}'.format(features))
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

    for i in dropList:
        if i[0] in features and i[1] in features:
            features.remove(i[1])
    #print('Selected features:{}'.format(features))

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



# def get_df_from_indList(df,l):

def train_LR(train_data, train_labels):

    lr_clf = LogisticRegression(max_iter=99999999999)
    lr_clf.fit(train_data.to_numpy(), train_labels)
    return lr_clf



import time
start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))


def test_k_fold(df, k,time_features=None):

    print("K_fold:"+str(k))
    print(df.index)
    data_samples=[i for i in range(df.shape[0])]
    labels=list(df.label)
    del df['label']
    # print(df)
    # return
    # print(labels)
    # print(data_samples)

    X = np.array(data_samples)  # create an array
    y = np.array(labels)  # Create another array
    kf = KFold(n_splits=k,shuffle=True)
    #print(kf.get_n_splits(X))
    print("Splitting iterations:{}".format(kf))
    df=df.reset_index(drop=True)
    df = StandardScaler().fit_transform(df)
    df = pd.DataFrame(data=df)

    #print(df)


    score_list = []
    precision_scores = []
    recall_scores = []

    score_list_train = []
    precision_scores_train = []
    recall_scores_train = []
    time_train = []
    time_test = []
    for train_index, test_index in kf.split(X):
        print('TRAIN Indexes' + str(len(train_index)) + '   TEST:' + str(len(test_index)))

        train_data=df[df.index.isin(train_index)]
        train_labels = [labels[l] for l in train_index]


        test_data = df[df.index.isin(test_index)]
        test_labels = [labels[l] for l in test_index]

        start = time.time()

        lr_clf = train_LR(train_data, train_labels)

        end=time.time() - start
        time_train.append(end)
        print("---Train time %s seconds ---" % (end))

        score_currunt = lr_clf.score(test_data, test_labels)

        start = time.time()
        y_pred = lr_clf.predict(test_data)
        end = time.time() - start
        precision_scores.append(precision_score(test_labels, y_pred, average='micro'))
        recall_scores.append(recall_score(test_labels, y_pred, average='micro'))
        score_list.append(score_currunt)



        print("Test_Score:{}".format(score_currunt))
        print("---Test time %s seconds ---" % (end))
        time_test.append(end)

        score_currunt = lr_clf.score(train_data, train_labels)
        y_pred = lr_clf.predict(train_data)
        precision_scores_train.append(precision_score(train_labels, y_pred, average='micro'))
        recall_scores_train.append(recall_score(train_labels, y_pred, average='micro'))
        score_list_train.append(score_currunt)
        print("Train_Score:{}".format(score_currunt))
        print('')

    print("**********Total Test Score:{}**********".format(np.mean(score_list)))
    print("Recall: %0.2f (+/- %0.2f)" % (np.mean(precision_scores), np.std(precision_scores) * 2))
    print("Recall: %0.2f (+/- %0.2f)" % (np.mean(recall_scores), np.std(recall_scores) * 2))
    print('\n')

    print("**********Total Train Score:{}**********".format(np.mean(score_list_train)))
    print("Recall: %0.2f (+/- %0.2f)" % (np.mean(precision_scores), np.std(precision_scores_train) * 2))
    print("Recall: %0.2f (+/- %0.2f)" % (np.mean(recall_scores), np.std(recall_scores_train) * 2))
    print("---Average train time %s seconds ---" % (sum(time_train)/len(time_train)))
    print("---Average Test time %s seconds ---" % (sum(time_test) / len(time_test)))

    with open('prediction_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["testing_accuracy", "time_get_features_and_train", "time_predict"])
        writer.writerow([np.mean(score_list), time_features+sum(time_train)/len(time_train), (sum(time_test) / len(time_test))])






from extract_features import *
def train():
    start = time.time()
    extract()
    end = time.time()

    features_list=load_features_list()
    print("Features:"+str(features_list))
    df=get_features_from_list(features_list[0:7])
    print(df)
    test_k_fold(df,5,end-start)



train()
