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

def main():
    os.chdir('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\data-scraper')
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
        data = pd.read_csv('combinedData.csv')

        #data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'teamER', 'ERA', 'pitchersUsed'], axis=1, inplace=True)
        data.drop(['League', 'teamAbbr', 'RBI', 'indER', 'pitchersUsed', 'teamER', 'Win'], axis=1, inplace=True)
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

        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]

        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks_list)

        plot_history(history)

        predictions = model.predict(X_test)
        print("Predictions: ", str(np.floor(predictions)))

        final_mse = mean_squared_error(y_test, predictions)
        print(final_mse)
        final_rmse = np.sqrt(final_mse)
        print(final_rmse)

        model.save('C:\\Users\\delevan\\PycharmProjects\\Senior-Project\\neural-net\\models\\regModel')

        ## --------------------------------------------------------------- ##

#def build_model(X_train):
#  model = keras.Sequential([
#    layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
#    layers.Dense(64, activation='relu'),
#    layers.Dense(1)
#  ])

#  optimizer = tf.keras.optimizers.RMSprop(0.001)

#  model.compile(loss='mse',
#                optimizer=optimizer,
#                metrics=['accuracy','mse'])
#  return model

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