import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical


def dataExploration(df, df2):
    print("\n== Dataframe description ==")
    print(df.describe())  # general description of the dataframe
    print("\n", df.head())  # print the first 5 lines
    print("\n", df.tail(3))  # print the last 3 lines

    # plot accel. data
    df[['acc2', 'acc3']].plot()
    plt.show()

    # plot sound data
    df['acc1'].plot()

    plt.show()

    # merge mulple dataset
    print("\n== Dataframe concatenation ==")

    print("\nDataframe description")
    print("\n", df2.describe())  # general description of the dataframe
    print("\n", df.head())  # print the first 5 lines

    result = pd.concat([df, df2], axis=1)  # concatenate 2 df (colum-wise)
    print("\nConcatenation results")
    print("\n", result.describe())
    print("\n", result.head())

    # convert dataframe to nparray
    print("\n== Conversion to numpy ==")
    X_train = df.to_numpy()
    print("Shape: ", X_train.shape)
    print(X_train[0])
    print(X_train[1])


if __name__ == "__main__":
    df = pd.read_csv("../data/orange_75_balanced_no_load_electric_fault.csv",
                     header=None, names=["acc1", "acc2", "acc3"])
    df2 = pd.read_csv("../data/orange_75_balanced_no_load.csv",
                      header=None, names=["acc1", "acc2", "acc3"])
    # dataExploration(df, df2)


    # Model / data parameters
    num_classes = 8


    motorSpeed = ['75','80','85','90']
    labels = ["balanced_no_load",
              "balanced_no_load_electric_fault",
              "balanced_with_load",
              "balanced_with_load_electric_fault",
              "unbalanced_no_load",
              "unbalanced_no_load_electric_fault",
              "unbalanced_with_load",
              "unbalanced_with_load_electric_fault"]


    y_train = []
    X_train_df = pd.DataFrame()
    for i in range(0, len(labels)-1):
        for j in motorSpeed :
            df = pd.read_csv("../data/orange_"+j+"_"+labels[i]+".csv", header=None, names=["acc1", "acc2", "acc3"])

            tmp = df.to_numpy()
            nbLine = int(tmp.shape[0] * 0.6)
            X_train_df = pd.concat([X_train_df, df.head(nbLine)], axis=0)
            for k in range(0,nbLine-1):
                y_train.append(i)

    X_train = X_train_df.to_numpy()


    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

