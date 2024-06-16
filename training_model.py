from loading_data import create_test_data, create_training_data

import tensorflow as tf
import pickle
import os

#Loading data
if os.path.exists("X_train.pickle") and os.path.exists("y_train.pickle"):
    print("LOADING TRAINING DATA")
    x_train = pickle.load(open("X_train.pickle", "rb"))
    y_train = pickle.load(open("y_train.pickle", "rb"))
else:
    create_training_data()
    print("LOADING TRAINING DATA")
    x_train = pickle.load(open("X_train.pickle", "rb"))
    y_train = pickle.load(open("y_train.pickle", "rb"))

#normalizing data
x_train = x_train / 255.0

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print(f"\nReached 99% accuracy so cancelling training!")

def training_model():
    
    #initialising teh callback class
    callbacks = myCallback()


    #building model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(tf.keras.layers.Dense(64, activation = 'relu'))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    
    #model Compilation
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=["accuracy"])

    #Fitting the model
    model.fit(x_train,y_train, batch_size=32, validation_split = 0.1, epochs = 5, callbacks=[callbacks])

    #Saving the model
    model.save("dots.keras")