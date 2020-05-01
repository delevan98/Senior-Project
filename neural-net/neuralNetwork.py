import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Flatten, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
import shutil

def main():
    '''
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\weights')
    teamAbbr = ["CHN", "PIT", "PHI", "CIN", "SLN", "BOS", "CHA",
                "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI",
                "MIL", "WAS"]

    ## NEURAL NETWORK CLASSIFIER ##

    #tf.debugging.set_log_device_placement(True)

    with tf.device('/GPU:0'):
        epochs = 1000
        learning_rate = .01
        data = pd.read_csv('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper\\combinedData.csv')

        #data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'ERA', 'pitchersUsed'], axis=1, inplace=True)
        data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'pitchersUsed', 'teamER', 'Win', 'wonPrev', 'OPS'], axis=1, inplace=True)
        data = data.astype(float)



        X_train, X_test, y_train, y_test = train_test_split(data.drop('Score', axis=1),
                                                            data['Score'], test_size=0.20,
                                                            random_state=101)

        def create_model(X_train):
            model = Sequential()

            # The Input Layer :
            model.add(Dense(24, kernel_initializer='normal', input_dim=X_train.shape[1], activation='relu'))

            # The Hidden Layers :
            model.add(Dense(12, kernel_initializer='normal', activation='relu'))
            model.add(Dense(12, kernel_initializer='normal', activation='relu'))
            model.add(Dense(12, kernel_initializer='normal', activation='relu'))

            # The Output Layer :
            model.add(Dense(1, kernel_initializer='normal', activation='linear'))

            return model

        model = create_model(X_train)

        # Compile the network :
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        model.summary()

        # checkpoint_name = 'Weights-best.hdf5'
        # checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        # callbacks_list = [checkpoint]

        # history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2) #, callbacks=callbacks_list)

        # plot_history(history)

        model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)
        predictions = model.predict(X_test)


        #print("Predictions: ", str(np.floor(predictions)))

        final_mse = mean_squared_error(y_test, predictions)
        #print(final_mse)
        final_rmse = np.sqrt(final_mse)
        print('NN RMSE:' + str(final_rmse))

        model.save('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\regModel')

        ## --------------------------------------------------------------- ##
        '''

    ## -------------------- Classification LSTM ---------------------- ##

    from statsmodels.tsa.stattools import adfuller, kpss
    data = pd.read_csv('D:\\Downloads\\projectBackup\\data-scraper\\ANA_Full.csv')
    data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER'], axis=1, inplace=True)

    time_steps = 10

    # reshape to [samples, time_steps, n_features]

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]
    print(len(train), len(test))

    X_train, y_train = create_dataset(train, train.Win, time_steps)
    X_test, y_test = create_dataset(test, test.Win, time_steps)

    print(X_train.shape, y_train.shape)

    model = Sequential()
    model.add(LSTM(units=24, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=12, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=12, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=12))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        shuffle=False
    )




def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def plot_history(history):
  print(history.history)
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  hist.tail(10)

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Root Mean Square Error [$Score^2$]')

  #plt.plot(hist['epoch'], hist['accuracy'], label='Train Accuracy')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,12])
  plt.legend()
  plt.show()

if __name__ == "__main__":
    main()