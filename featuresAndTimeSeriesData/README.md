# dataMining

####Requirements
```
$ pip3 install numpy pandas seaborn sklearn
```


######Plot orignal time series data


```
$ python3.6 extract.py -m orignal

```
###### Plot the mean of all the data samples

```
$ python3.6 extract.py -m mean -p plot
```

###### For Mean, extract the independent features using corelation, and display the similarity matrix and plot the all the mean features from each time series


```
$ python3.6 extract.py -m mean -f extract -e head

```

###### For Standard Deviation, extract the independent features using corelation, and display the similarity matrix and plot the all the std features from each time series


```
$ python3.6 extract.py -m std -f extract -p plot -e head

```

###### For Max, extract the independent features using corelation, and display the similarity matrix and plot the all the max features from each time series

```
$ python3.6 extract.py -m max -f extract -p plot -e head

```

###### For Min, extract the independent features using corelation, and display the similarity matrix and plot the all the min features from each time series

```
$ python3.6 extract.py -m min -f extract -p plot -e head

```

###### Creata a Feature matrix from all the independent variables

```
$ python3.6 extract.py -m matrix

```
###### Feed the Feature Matrix to PCA, Select the top five features and plot it

```
$ python3.6 extract.py -m pca 

```
