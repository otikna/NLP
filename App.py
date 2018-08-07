import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import itertools
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import keras
import pickle
import tensorflow as tf
import keras

# config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

# from tensorflow.python.client import device_lib
#
# print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# PreProcessing
# df = pd.read_csv("YemekSepeti.csv", sep=';', index_col=0)
# df = df[['score_flavour', 'score_serving', 'score_speed', 'text_']]
# df['Score'] = round((df['score_flavour'] + df['score_serving'] + df['score_speed']) / 3)
# dummyScore = []
# for i in np.array(df[['Score']]):
#     if int(i) < 5:
#         dummyScore.append("Olumsuz")
#     elif int(i) < 7:
#         dummyScore.append("Nötr")
#     else:
#         dummyScore.append("Olumlu")
#
# dataFrame = pd.DataFrame(data=df[['text_']]).join(pd.DataFrame(data=dummyScore, columns=["Score"]))
# dataFrame['text_'] = dataFrame['text_'].apply(lambda x: x.lower())
# dataFrame['text_'] = dataFrame['text_'].apply(lambda x: re.sub('[^a-zA-z0-9üğçöış\s]','',x))
# dataFrame['text_'] = dataFrame['text_'].apply(lambda x: ''.join(c[0] for c in itertools.groupby(x)))
# dataFrame.to_csv("PreProcess.csv")


dataFrame = pd.read_csv("PreProcess.csv", index_col=0)
#
#
# tokenizer = Tokenizer(split=' ')
# tokenizer.fit_on_texts(dataFrame['text_'].values.astype(str))
#
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Train Test Split
X = pad_sequences(tokenizer.texts_to_sequences(dataFrame['text_'].values.astype(str)), maxlen=60)
Y = pd.get_dummies(dataFrame['Score'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# #Build Model--KFold Ekle
# #
# model = keras.Sequential()
# model.add(keras.layers.Embedding(1000000, 32, input_length=X.shape[1]))
# model.add(keras.layers.SpatialDropout1D(0.4))
# model.add(keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(keras.layers.Dense(512, activation='sigmoid'))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Dense(256, activation='sigmoid'))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Dense(3, activation='sigmoid'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
#
# model.fit(X_train, Y_train, 64, epochs=1)
# model.save("modelSave")

model = keras.models.load_model("modelSave")
pred = model.predict(X_test)
for i in pred:
    max = np.array(i).max()
    for idx, j in enumerate(i):
        if j == max:
            i[idx] = 1
        else:
            i[idx] = 0

from sklearn.metrics import accuracy_score

print('Modelin Doğruluk Oranı -> {0:.2F}'.format(accuracy_score(Y_test, pred)))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(Y_test.values.argmax(axis=1), pred.argmax(axis=1)))
