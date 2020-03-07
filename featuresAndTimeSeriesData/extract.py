import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

sns.set()
from datetime import datetime
import matplotlib.pyplot as plt

import argparse




def getfeaturesfromDf(df1,head):

    colsSeries=[]
    for i in range(1, 32):
        colsSeries.append('cgmSeries_' + str(i))

 #   print (df[0])
    cor = df1[colsSeries].corr()

    #cor = df1.corr()
   # print (cor)
   # sns.set()
    if( not(head==None)):
        plt.figure(figsize=(20, 20))
        plt.title('Correlation Matrix')
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()

    cols = df1.columns
    cor_target = abs(cor[cols[31]])

    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0.5]
 #   relevant_features.columns=['A','B']


    potentialfeaturs=relevant_features.index
    print('Potential relevent feature related to:' + cols[31])
    print(potentialfeaturs)
   # print (df.columns)
    dropList=[]
    for i in range(1,len(potentialfeaturs)):
        for j in range(1, len(potentialfeaturs)):

            if(not(potentialfeaturs[i]== potentialfeaturs[j])):
                indcor=df1[[potentialfeaturs[i], potentialfeaturs[j]]].corr()
                indcor = abs(indcor[potentialfeaturs[i]])
                #print (indcor[1])
                if(indcor[1]>.9):
                    if(not(potentialfeaturs[i] in dropList)):
                       # print ('Comparing independent varibles:{} and {}, Corr score Is {}; dropping : {}'.format(potentialfeaturs[i], potentialfeaturs[j],indcor[1],potentialfeaturs[j]))
                        dropList.append(potentialfeaturs[j])


    dropList= set(dropList)

    print ("Independable variable thae are have strong correllation, Droping:{}".format(dropList))
    features=list(set(potentialfeaturs)-dropList)
   # features.append(cols[0])
    print ("Using {} Selected Features are :{} ,".format('',features))
    return features

                # print( df[[potentialfeaturs[i],potentialfeaturs[j]]].corr())


    #print(relevant_features.index)



def showorignaltimeSeriesWithMean():
  fig, ax = plt.subplots(nrows=5, ncols=1,figsize=(20,20))
  sub=0;
  for row in ax:
    sub=sub+1
    colsTime = []
    colsSeries = []


    dfTime = pd.read_csv('CGMDatenumLunchPat' + str(sub) + '.csv', skiprows=1)
    dfSeries = pd.read_csv('CGMSeriesLunchPat' + str(sub) + '.csv', skiprows=1)
    for i in range(1, dfSeries.shape[1] + 1):
        colsTime.append('cgmDatenum_' + str(i))
        colsSeries.append('cgmSeries_' + str(i))

    dfSeries.columns = colsSeries
   # print(dfSeries.head())
    #getfeaturesfromDf(dfSeries)

    # dfSeries=dfSeries[colsSeries]
    # print (dfSeries.head())
    ##dfTime=dfTime[colsTime]
    for i in colsSeries:
        dfSeries[i] = dfSeries[i].rolling(3).mean()
    df = pd.concat([dfTime, dfSeries], axis=1, sort=False)
    cols = colsTime + colsSeries
    df.columns = cols
    args = []
    for i in range(len(colsSeries)):
        args.append(df[colsTime[i]])
        args.append(df[colsSeries[i]])

    row.plot(*args)

     #plt.figure(figsize=(20, 20), dpi=70)
  plt.title("Orignal Data, 5 subjects")
  plt.legend(colsSeries,loc='center left',bbox_to_anchor=(1, 2))
  plt.xlabel('Time', fontsize=10);
  plt.show()




def check1(a,  b):
        if (a-10< b <a+10):
            return True
        return False


def CombineSubjects():
  #fig, ax = plt.subplots(nrows=5, ncols=1)
  sub=0;
  dfTime=[]
  dfSeries=[]
  dflist=[]
  for row in range(5):
    sub=sub+1
    colsTime = []
    colsSeries = []



    dfTime.append( pd.read_csv('CGMDatenumLunchPat' + str(sub) + '.csv', skiprows=1))
    dfSeries.append(pd.read_csv('CGMSeriesLunchPat' + str(sub) + '.csv', skiprows=1))
    for i in range(1, dfSeries[sub-1].shape[1] + 1):
        colsTime.append('cgmDatenum_' + str(i))
        colsSeries.append('cgmSeries_' + str(i))

    dfSeries[sub-1].columns = colsSeries
    dfTime[sub-1].columns=colsTime
   # print(dfSeries.head())
    #getfeaturesfromDf(dfSeries)

    # dfSeries=dfSeries[colsSeries]
    # print (dfSeries.head())
    ##dfTime=dfTime[colsTime]
    for i in colsSeries:
        dfSeries[sub-1][i] = dfSeries[sub-1][i].rolling(3).mean()
    for i in colsTime:
       # dfTime[sub - 1][i] = dfTime[sub - 1][i].rolling(3).mean()
        dfTime[sub - 1][i]=round(dfTime[sub - 1][i])



    df = pd.concat([dfTime[sub-1], dfSeries[sub-1]], axis=1, sort=False)
    cols = colsTime + colsSeries
    df.columns = cols
    dflist.append(df)




  finalDf=getjoin(dflist)
  return finalDf;

 # dflist[1].set_index('cgmDatenum_1', inplace=True)
 # dflist[0].set_index('cgmDatenum_1', inplace=True)
  #
  #
  # for tt in range(1,31):
  # print(getjoin(dflist)['cgmSeries_'+str(tt)+'_2'])
  #print( dflist[0])
 # print (dflist[0].shape[1])

 # combineDf=pd.DataFrame()
  #for dfcol in dflist[0].columns:
   #   for dfrow in range(dflist[0].shape[0]):
    #       print(dflist[0][dfcol][dfrow])
     #      if(dflist[0][dfcol][dfrow]==None or dflist[1][dfcol][dfrow]==None or dflist[2][dfcol][dfrow]==None or dflist[3][dfcol][dfrow]==None or dflist[4][dfcol][dfrow]==None):
      #         combineDf[dfcol][dfrow]=dflist[0][dfcol][dfrow]
       #    else:
        #       print(dflist[0][dfcol][dfrow]+dflist[1][dfcol][dfrow] + dflist[2][dfcol][dfrow]+ dflist[3][dfcol][dfrow]+dflist[4][dfcol][dfrow])
              # print(combineDf[dfcol][dfrow])




  #plt.figure(figsize=(5, 5), dpi=70)

    # plt.plot(*args)
    # plt.legend(colsSeries)
  plt.xlabel('Year', fontsize=20);
  plt.show()


def getjoin(dflist):
  for i in range(1,5):
    sh1 = dflist[0].shape[0]
    sh2 = dflist[i].shape[0]
    joincols=[]


    for ii in range(1, 32):
        joincols.append('cgmSeries_' + str(ii)+'_'+str(i))
   # print(joincols)
    for ccc in joincols:
          dflist[0][ccc] = np.nan
    datJ1 = dflist[0]['cgmDatenum_1'][dflist[0].shape[0] - 1]
    datJ2 = dflist[i]['cgmDatenum_1'][dflist[i].shape[0] - 1]
    prev=None
    for iJ in range(sh1):
        for jJ in range(sh2):

            if ( check1(dflist[0]['cgmDatenum_1'][iJ] - datJ1, dflist[i]['cgmDatenum_1'][jJ] - datJ2)):


                if (not (dflist[i]['cgmDatenum_1'][jJ] == prev)):
                    prev = dflist[i]['cgmDatenum_1'][jJ]
                    bb=False
                    for zz in joincols:
                      bb=True
                      dflist[0][zz][iJ] = dflist[i][zz[0:len(zz)-2]][jJ]
                      #print(  dflist[0][zz][iJ] )
                    if(bb==True):
                     break

  return dflist[0];
def getargs(dff):

    colsTime = []
    colsSeries = []

    for i in range(1, 32):
        colsTime.append('cgmDatenum_' + str(i))
        colsSeries.append('cgmSeries_' + str(i))

    #print(dff['cgmDatenum_1'])


   # dfSeries=dfSeries[colsSeries]
   # print (dfSeries.head())
    ##dfTime=dfTime[colsTime]
    arg1s=[]

    for iii in range(len(colsSeries)):
       # print(dff[colsTime[iii]])
        arg1s.append(dff[colsTime[iii]])
        arg1s.append(dff[colsSeries[iii]])
    return arg1s



#print (pd.Timestamp('737225.584155093',unit='s'))

#exit()
#dfTime = pd.read_csv('CGMDatenumLunchPat4.csv', skiprows=1)
#dfSeries = pd.read_csv('CGMSeriesLunchPat4.csv', skiprows=1)
#print(dfSeries)
#exit(0)
#ShoworignaltimeSeriesWithMean()

def mean(dff,method):


    colsTime = []
    colsSeries = []

    colsss=[]
    for i in range(1, 32):
      #  colsss.append('cgmDatenum_' + str(i))
        colsss.append('cgmDatenum_' + str(i))
    for i in range(1, 32):
           # colsTime.append('cgmDatenum_' + str(i))
            colsss.append('cgmSeries_' + str(i))
    for i in range(1, 32):

        colsTime.append('cgmDatenum_' + str(i))
        colsSeries.append('cgmSeries_' + str(i))
    retdf = pd.DataFrame(columns=colsss)
  #  print(retdf.columns)
    count=0;
    for ccc in colsss:
       retdf[ccc]=dff[ccc]
   # print(retdf['cgmSeries_3'])



    for rrow in range(dff.shape[0]):
        #  print (rrow)
          for cccol in colsSeries:
              # print("{} {} {} {} {}1".format(dff[cccol][rrow], dff[cccol + '_1'][rrow], dff[cccol + '_2'][rrow],
               #                             dff[cccol + '_3'][rrow], dff[cccol + '_4'][rrow]))
             #  if(not(np.isnan(dff[cccol+'_1'][rrow]) or np.isnan(dff[cccol+'_2'][rrow]) or np.isnan(dff[cccol+'_3'][rrow]) or np.isnan(dff[cccol+'_4'][rrow]) )):
                #       print ("{} {} {} {} {}".format(dff[cccol][rrow],dff[cccol+'_1'][rrow],dff[cccol+'_2'][rrow],dff[cccol+'_3'][rrow],dff[cccol+'_4'][rrow]))
               newL=[dff[cccol][rrow] , dff[cccol + '_1'][rrow], dff[cccol + '_2'][rrow], dff[cccol + '_3'][rrow], dff[cccol + '_4'][rrow]]
               cleanedList = [x for x in newL if not str(x) == 'nan']
               #print("ll{}".format(cleanedList))

               if(len(cleanedList)>0):
                   if(method=='mean'):
                        retdf[cccol][rrow]=int(np.mean(cleanedList))
                   if (method == 'min'):
                       retdf[cccol][rrow] = int(np.min(cleanedList))
                   if (method == 'max'):
                       retdf[cccol][rrow] = int(np.max(cleanedList))
                   if (method == 'std'):
                       retdf[cccol][rrow] = int(np.std(cleanedList))
             #  print (  "aaa{}".format(retdf[cccol][rrow]))
                      # print(retdf[cccol][rrow])
              # else:
               #    retdf[cccol]= dff[cccol][rrow]
   # print(retdf['cgmSeries_3'])
   # print(retdf.columns)
   # print (type(retdf))
    return retdf








def myPlot(df,title):

    colsTime = []
    colsSeries = []

    for i in range(1, 32):
        colsTime.append('cgmDatenum_' + str(i))
        colsSeries.append('cgmSeries_' + str(i))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    ax.plot(*getargs(df))

    #plt.plot(*getargs(df))
   # plt.figure(figsize=(20, 20), dpi=70)


    plt.title(title)
    #ax.set_ylabel('Sensor Values')
    plt.legend(colsSeries, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time', fontsize=10);
    plt.show()



def myPlot1(df,title,l):

    colsTime = []
    colsSeries = []

    for i in range(1, 32):
        colsTime.append('cgmDatenum_' + str(i))
        colsSeries.append('cgmSeries_' + str(i))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))

    arg1ss=[]
    for iii in range(len(df.columns)):
       # print(dff[colsTime[iii]])
        arg1ss.append(l);
        arg1ss.append(df[df.columns[iii]])


    ax.plot(*arg1ss)

    #plt.plot(*getargs(df))
   # plt.figure(figsize=(20, 20), dpi=70)

    # plt.plot(*args)
    plt.title(title)
    #ax.set_ylabel('Sensor Values')
    plt.legend(df.columns, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time', fontsize=10);
    plt.show()



from sklearn.decomposition import PCA

def getpca(head,pcaArg):

    colsTime = []
    colsSeries = []
    for i in range(1, 32):

        colsTime.append('cgmDatenum_' + str(i))
        colsSeries.append('cgmSeries_' + str(i))
    dfMax = mean(CombineSubjects(), 'max')
    dfMin = mean(CombineSubjects(), 'min')
    dfStd = mean(CombineSubjects(), 'std')
    dfMean = mean(CombineSubjects(), 'mean')

    dffMin = dfMin[getfeaturesfromDf(dfMin, head)]

    dffMean = dfMean[getfeaturesfromDf(dfMean, head)]
    dffStd = dfStd[getfeaturesfromDf(dfStd, head)]
    dffMax = dfMax[getfeaturesfromDf(dfMax, head)]
    dfC = pd.concat([dffMean, dffStd, dffMax, dffMin], axis=1, sort=False)
    comCol = [col + '_mean' for col in dffMean.columns] + [col + '_std' for col in dffStd.columns] + [col + '_max'
                                                                                                           for col in
                                                                                                           dffMax.columns] + [
                      col + '_min' for col in dffMin.columns]
    dfC.columns=comCol
    origDf=dfC[comCol]
    origDf.columns = comCol
    #dfC = dfC[pd.notnull(dfC)]
    dfC.dropna( how='all', inplace=True)
    dfC.to_csv('derivedMatrix.csv', index=False)
    X_std=[dfC[comCol]]
    print("MATRIX FOR PCA---->{}".format(dfC))

    if(pcaArg=='pca'):


        pca = PCA(n_components=2)
        pca.fit(dfC)


        columns = ['pca_%i' % i for i in range(2)]
        df_pca = DataFrame(pca.transform(dfC), columns=columns, index=dfC.index)
        print(df_pca)
        #pca_score = pca.explained_variance_ratio
        comp=pca.components_
        comp1=comp[0]

        #print ('FirstPricipalComponent:{}'.format(comp1))
        comp1 = np.abs(comp1)
        print ('Fisrt Pricncpal Component{}'.format(comp1))
        comp1=np.abs(comp1)
        top=comp1.argsort()[-5:][::-1]
        print (top)
        topFeatures=[]
        for tt in top:
            print ('Top 5 Featues from PCA={}'.format(comCol[tt]))
            topFeatures.append(comCol[tt])
        tempF= origDf[topFeatures]
        tempF.columns=topFeatures
        print('Derived feature matrix from pca{}:'.format(tempF))
       # derivedFM = pd.concat([ dfMax[colsTime], tempF], axis=1, sort=False)
        #comFCol=colsTime+topFeatures
        #derivedFM.columns=comFCol
        myPlot1(tempF,"Feature selected By Pca",dfMax['cgmDatenum_1'])
        dfv = pd.DataFrame({'var': pca.explained_variance_ratio_,
                           'PC': ['PC1', 'PC2']})
        sns.barplot(x='PC', y="var",
                    data=dfv, color="c");
        plt.show()
       

    # print(len(dffMax))
    # print(len(dffMin))
    # print(len(dffMean))
    # print(len(dffMax))


def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0
        index=0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];
                index=j

        list1.remove(max1);
        final_list.append(max1)

    print(final_list)


# Driver cod




parser = argparse.ArgumentParser(description='Assingment 1')
parser.add_argument('-m', '--method', help='orignal/mean/min/max/std', required=False)
parser.add_argument('-f', '--features', help='extract', required=False)
parser.add_argument('-p', '--plot', help='show', required=False)
parser.add_argument('-e', '--head', help='show', required=False)
parser.add_argument('-r', '--run', help='Run PCA', required=False)
cmdargs = parser.parse_args()
print(cmdargs)


if(cmdargs.method=='orignal'):
    showorignaltimeSeriesWithMean();
if(cmdargs.method=='mean'):
    df = mean(CombineSubjects(), 'mean')
    #print("Mean DataFrame:{}".format(df[0].head()))
    #df[0].to_csv('mean.csv',index=False)
    # print(type(df))

    if (not (cmdargs.features == None)):
        dfNew = df[getfeaturesfromDf(df,cmdargs.head)]
        dfNew.to_csv('mean.csv', index=False)
        print("Mean DataFrame:{}".format(dfNew.head()))

    if (not (cmdargs.plot == None)):
        myPlot(df, 'Each point is the mean of all the 5 Samples')
if (cmdargs.method == 'min'):
    df = mean(CombineSubjects(), 'min')
   # df[0].to_csv('min.csv', index=False)
  #  print("Min DataFrame:{}".format(df[0].head()))

    if (not (cmdargs.features == None)):
        dfNew = df[getfeaturesfromDf(df,cmdargs.head)]
        dfNew.to_csv('min.csv', index=False)
        print("min DataFrame:{}".format(dfNew.head()))

    if (not (cmdargs.plot == None)):
        myPlot(df, 'Each point is the Min of all the 5 Samples')

if (cmdargs.method == 'max'):
    df = mean(CombineSubjects(), 'max')
   # df[0].to_csv('max.csv', index=False)

    if (not (cmdargs.features == None)):
        dfNew=df[getfeaturesfromDf(df,cmdargs.head)]
        dfNew.to_csv('max.csv', index=False)
        print("Max DataFrame:{}".format(dfNew.head()))


    if (not (cmdargs.plot == None)):
         myPlot(df, 'Each point is the Max of all the 5 Samples')
if (cmdargs.method == 'std'):
    df = mean(CombineSubjects(), 'std')


    if (not (cmdargs.features == None)):
        dfNew = df[getfeaturesfromDf(df,cmdargs.head)]
        dfNew.to_csv('std.csv', index=False)
        print("StD DataFrame:{}".format(dfNew.head()))

    if (not (cmdargs.plot == None)):
        myPlot(df, 'Each point is the Standart deviation of all the 5 Samples')


if (cmdargs.method == 'matrix'or cmdargs.method == 'pca'):
    getpca(cmdargs.head,cmdargs.method)