import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle


X_train = np.load('model/X.txt.npy')
Y_train = np.load('model/Y.txt.npy')
print(X_train.shape)
print(Y_train.shape)
if os.path.exists('model/cnnmodel.json'):
    with open('model/cnnmodel.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/cnnmodel_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/cnnhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
else:
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=30, shuffle=True, verbose=2)
    classifier.save_weights('model/cnnmodel_weights.h5')            
    model_json = classifier.to_json()
    with open("model/cnnmodel.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/cnnhistory.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/cnnhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Face Model Accuracy = "+str(accuracy))
