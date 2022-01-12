import datetime

import numpy as np
import pandas as pd
import csv
import ctypes

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from owlready2 import *
from decimal import Decimal

# from datetime import datetime

# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout


def SearchAnomaly(df):
    split = 0.5

    cutoff = int(len(df) * split)
    train_df = df.head(cutoff)
    test_df = df.tail(len(df) - cutoff)

    training_mean = train_df.mean()
    training_std = train_df.std()
    df_training_value = (train_df - training_mean) / training_std

    df_test_value = (test_df - training_mean) / training_std
    # fig, ax = plt.subplots()
    # df_test_value.plot(legend=False, ax=ax)
    # plt.show()

    # print(df_training_value)

    TIME_STEPS = 12

    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i: (i + time_steps)])
        print(len(values))
        return np.stack(output)

    # print(df_training_value.values)
    # print(df_test_value.values)

    x_train = create_sequences(df_training_value.values)
    print("Training input shape: ", x_train.shape)

    x_test = create_sequences(df_test_value.values)
    print("Test input shape: ", x_test.shape)

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
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same", batch_size=50),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=50,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.legend()
    # plt.show()

    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    # plt.hist(train_mae_loss, bins=50)
    # plt.xlabel("Train MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)

    # plt.plot(x_train[0])
    # plt.plot(x_train_pred[0])
    # plt.show()

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    # plt.hist(test_mae_loss, bins=50)
    # plt.xlabel("test MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    print("Number of anomaly samples: ", np.sum(anomalies))
    print("Indices of anomaly samples: ", np.where(anomalies))

    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):

        # print(np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]))
        if np.all(anomalies[data_idx - TIME_STEPS + 1: data_idx]):
            anomalous_data_indices.append(data_idx)
    df_subset = test_df.iloc[anomalous_data_indices]
    # print(df_subset)

    return df_subset
    # plt.show()


def UsingOntology(df_anomaly):
    # onto_path.append("C:/Users/Dimon/PycharmProjects/protorype/")
    # onto = get_ontology("C://Users/Dimon/PycharmProjects/protorype/prototype.owl").load()
    onto = get_ontology("prototype.owl").load()

    for individual in onto.individuals():
        destroy_entity(individual)
    # print(df_anomaly)

    with onto:
        # class Well2(Thing):
        #     namespace = onto
        #     pass
        #
        # class hasWell_Name2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [str]
        #
        # class hasDate2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [str]
        #
        # class hasDeltaPHI2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [float]
        #
        # class hasDepth2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [float]
        #
        # class hasGR2(DataProperty, FunctionalProperty):
        #     domain = [Well2]
        #     range = [float]

        for index, anomaly in df_anomaly.iterrows():
            # print(str(index))
            well = onto.Well(
                hasDate=str(index),
                hasWell_Name=anomaly['Well Name'])
            #print(Decimal(anomaly['DeltaPHI']*1000))
            well.hasDeltaPHI = (anomaly['DeltaPHI'])
            well.hasDepth = (anomaly['Depth'])
            well.hasGR = (anomaly['GR'])
            #print(onto.hasDepth.range)

        #sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=3)
        onto.save(file="prototypePython.owl", format="rdfxml")

    ctypes.windll.user32.MessageBoxW(0, "Запустите правила", "Пауза", 1)

    onto2 = get_ontology("prototypePython.owl").load()
    #print(df_anomaly.index)
    anomalous_data_indices = []
    for SWRLanomaly in onto2.individuals():
        if SWRLanomaly.hasAnomaly:
            #print(pd.to_datetime(SWRLanomaly.hasDate, format='%Y-%m-%d') unit='s'))

            anomalous_data_indices.append(pd.to_datetime(SWRLanomaly.hasDate, format='%Y-%m-%d'))
    #print(anomalous_data_indices.index)
    df_subset = df_anomaly.loc[anomalous_data_indices]
    #onto.save(file="prototypePython.owl", format="rdfxml")

    return df_subset


def printDF(df):
    plt.ion()

    dfNames = pd.DataFrame(pd.unique(df['Well Name']))

    dfNames = dfNames.drop(np.where(dfNames[0] == 'Recruit F9')[0]).to_numpy()
    # print(dfNames)
    # ['CHURCHMAN BIBLE' 'CROSS H CATTLE' 'LUKE G U' 'NEWBY' 'NOLAN' 'Recruit F9' 'SHANKLE' 'SHRIMPLIN']

    # plt.figure(1, figsize=(20, 10))
    # plt.subplots_adjust(wspace=0, hspace=1)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    i = 1
    for Name in dfNames[5:6]:
        Name = Name[0]
        dataForName = df.loc[df['Well Name'] == Name]
        dataForName = dataForName.sort_values("Depth")

        fig, axex = plt.subplots(nrows=1, ncols=1)

        # axex.plot(dataForName.index, dataForName["PHIND"], label="PHIND")

        # axex.plot(dataForName.index, dataForName["GR"], label="GR")

        # axex.plot(dataForName.index, dataForName["ILD_log10"], label="ILD_log10")

        #axex.plot(dataForName.index, dataForName["DeltaPHI"], label="DeltaPHI")
        # axex.legend()

        #plt.show()

        # plt.plot(data["date"], data["DeltaPHI"], label=Name)
        # plt.figure(i, figsize=(5, 5))

        # plt.subplot(1, 9, i)

        df_subset = SearchAnomaly(dataForName[["GR"]])
        #print(df_subset)
        plt.subplots_adjust(wspace=0, hspace=i)
        dataForName[["GR"]].plot(ax=axex, legend=False, color="black")
        if len(df_subset) > 0:
            df_subset.plot(ax=axex, legend=False, color="r", marker='s', linewidth=0)

        df_subset = dataForName.loc[df_subset.index]
        owl_ontology = UsingOntology(df_subset)
        #print(owl_ontology)
        if len(owl_ontology) > 0:
            owl_ontology["GR"].plot(ax=axex, legend=False, color="b", marker='o', linewidth=0)

        i += 1
    plt.ioff()
    plt.show()


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

# print(dfSource)
printDF(dfSource)

