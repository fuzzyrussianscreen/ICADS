import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
import csv

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt


from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def BuildModel(df):

    split = 0.75

    cutoff = int(len(df) * split)
    train_df = df.head(cutoff)
    test_df = df.tail(len(df)-cutoff)

    #print(train_df)
    #print()
    #print(test_df)

    training_mean = train_df.mean()
    training_std = train_df.std()
    df_training_value = (train_df - training_mean) / training_std

    #print(df_training_value)

    TIME_STEPS = 288

    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i: (i + time_steps)])
        return np.stack(output)


    x_train = create_sequences(df_training_value.values)
    print("Training input shape: ", x_train.shape)

    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()








def printDF(df):
    plt.ion()

    dfNames = pd.DataFrame(pd.unique(df['Well Name']))

    dfNames = dfNames.drop(np.where(dfNames[0] == 'Recruit F9')[0]).to_numpy()
    # print(dfNames)
    # ['CHURCHMAN BIBLE' 'CROSS H CATTLE' 'LUKE G U' 'NEWBY' 'NOLAN' 'Recruit F9' 'SHANKLE' 'SHRIMPLIN']

    plt.figure(1, figsize=(20, 10))
    plt.subplots_adjust(wspace=0, hspace=1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    for Name in dfNames[:1]:
        Name = Name[0]
        dataForName = df.loc[df['Well Name'] == Name]
        dataForName = dataForName.sort_values("Depth")

        # plt.subplot(2, 2, 1)
        # plt.plot(data["date"], data["Depth"], label=Name)

        #plt.subplot(1, 1, 1)
        #plt.plot(dataForName.index, dataForName["GR"], label=Name)

        # plt.subplot(2, 2, 3)
        # plt.plot(data["date"], data["ILD_log10"], label=Name)

        # plt.subplot(2, 2, 4)
        # plt.plot(data["date"], data["DeltaPHI"], label=Name)

        BuildModel(dataForName[["GR"]])

    plt.ioff()
    #plt.show()


# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.expand_frame_repr = False

with open('facies_data.csv', newline='') as csvfile:
    date = datetime.date.today()
    dfSource = pd.DataFrame(csv.reader(csvfile, delimiter=',', quotechar='|'))
    # print(df.columns)
    # ['Facies','Formation','Well Name','Depth','GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS','date']
    dfSource.columns = dfSource.iloc[0]
    dfSource.drop([0], inplace=True)
    dfSource = dfSource.reset_index(drop=True)
    dfSource["date"] = pd.date_range(start=date, end=date + datetime.timedelta(days=len(dfSource) - 1), freq='D')
    dfSource['date'] = pd.to_datetime(dfSource['date'], format='%Y-%m-%d %H:%M:%S')
    dfSource = dfSource.set_index('date')


    dfSource['Depth'] = dfSource['Depth'].astype(np.float)
    dfSource['DeltaPHI'] = dfSource['DeltaPHI'].astype(np.float)
    dfSource['ILD_log10'] = dfSource['ILD_log10'].astype(np.float)
    dfSource['GR'] = dfSource['GR'].astype(np.float)
    dfSource['Well Name'] = dfSource['Well Name'].astype(str)

#print(dfSource)
printDF(dfSource)
