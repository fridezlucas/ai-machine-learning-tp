import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn import model_selection
from sklearn import preprocessing


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

    motorSpeed = ['75', '80', '85', '90', '95', '100']
    labels = ["balanced_no_load",
              "balanced_no_load_electric_fault",
              "balanced_with_load",
              "balanced_with_load_electric_fault",
              "unbalanced_no_load",
              "unbalanced_no_load_electric_fault",
              "unbalanced_with_load",
              "unbalanced_with_load_electric_fault"]

    y_train = []
    # X_train_df = pd.DataFrame()

    y_test = []
    # X_test_df = pd.DataFrame()
    X = []
    y = []

    for i in range(0, len(labels)):
        for j in motorSpeed:
            try:
                df = pd.read_csv(
                    "../data/orange_"+j+"_"+labels[i]+".csv", header=None, names=["acc1", "acc2", "acc3"])
                tmp = df.to_numpy()
                window_size = 100
                for k in range(0, df.shape[0]-window_size*2, window_size):
                    ac1 = df[k:k+window_size]['acc1']
                    ac2 = df[k:k+window_size]['acc2']
                    ac3 = df[k:k+window_size]['acc3']
                    X.append([min(ac1), min(ac2), min(ac3),
                              max(ac1), max(ac2), max(ac3),
                              ac1.mean(),ac2.mean(),ac3.mean(),
                              ac1.std(),ac2.std(),ac3.std(),
                              ac1.quantile(.25),ac2.quantile(.25),ac3.quantile(.25),
                              ac1.quantile(.50),ac2.quantile(.50),ac3.quantile(.50),
                              ac1.quantile(.75),ac2.quantile(.75),ac3.quantile(.75),
                              ])
                    y.append(i)
            except Exception as e:
                print(e)

    X = np.array(X)
    y = np.array(y)
    X_train,  X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=True, shuffle=True)

    # Scaling the dataset
    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)


    model = Sequential([
        Dense(100, activation='relu', input_shape=(21,)),
        Dense(40, activation='relu'),
        Dense(8, activation='softmax'),
    ])

    model.summary()

    # Compile the model (configures the model for training).
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    print(X)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    # Train the model.
    history = model.fit(
        X_train,
        to_categorical(y_train),
        epochs=100,
        batch_size=256,
        validation_data=(X_test, to_categorical(y_test))
    )

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Evaluate the model.
    model.evaluate(
    X_test,
    to_categorical(y_test),
    verbose=2)

    # Save the model to disk.
    model.save_weights('model.h5')