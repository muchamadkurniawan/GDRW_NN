import pandas as pd
import pandas as pd
import numpy as np
from numpy import genfromtxt

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    df = genfromtxt('datasetNN.csv', delimiter=';', skip_header=1)
    data = (df[:,0:21])
    data =np.array(data).tolist()
    df1 = pd.read_excel(open('labelGLRW.xlsx', 'rb'))
    y = pd.DataFrame(df1, columns=(['label']))
    label = np.array(y).tolist()
    # print(len(data[0]))
    # for i in range (0,len(data)):
    #     data[i].append(y)
    #     y=y+24
    #
    # dataset =[]
    # for i in range(0, 175):
    #     for j in range(0,len(data)):
    y = 4300
    data=[]
    data1 = []
    for i in range (1000,300,-4):
        data.append(i)
    for i in range (175,0,-1):
        for j in range (4300,7000,24):
            data1.append([data[i-1], j])
    # print(data1)
    x = [8.000,2154.914,6.744,8.00001640,661.48551718,113.03031200,1777.625,377.28885358,5.564,1.181,0.006,0.000,6236.88,300.000,4300.000]
    # print(x)
    for i in range (0,len(data1)):
        for j in range(0,len(x)):
            data1[i].append(x[j])
    data2 = np.array(data1)
    # print(data2.shape)
    # print(label)

    # input_shape = data2.shape[1]
    arr = np.array(data1)
    input_shape = arr.shape[1]
    # print(arr.shape)
    print(len(data1))
    print(len(label))


    #
    # regr = MLPRegressor(random_state=1, max_iter=500).fit(data1, label)
    # regr.score(data2, label)

    model = tf.keras.Sequential([
        keras.layers.Dense(17,activation='relu', input_shape=(input_shape, )),
        # keras.layers.Dense(9, activation='relu'),
        keras.layers.Dense(1, activation='linear'),
        # keras.layers.Dense(16, activation='linear'),
    ])
    # Output lay
    model.compile(optimizer='adam',  # Use the Adam optimizer
                 loss='mae')  # Mean squared error loss for regression

    model.summary()
    model.fit(data1, label, epochs=400)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
