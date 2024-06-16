import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle
from category import TEST_CATEGORY, DATADIR_TRAIN, DATADIR_TEST

CATEGORIES_TRAIN = ["Dog", "Cat"]
CATEGORIES_TEST = TEST_CATEGORY

training_data = []
test_data = []

IMG_SIZE = 50

def create_training_data():
    print("CREATING TRAINING DATA")
    for category in CATEGORIES_TRAIN:  # do dogs and cats
        path = os.path.join(DATADIR_TRAIN,category)  # create path to dogs and cats
        class_num = CATEGORIES_TRAIN.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
    random.shuffle(training_data)

    X_train = []
    y_train = []

    for features,label in training_data:
        X_train.append(features)
        y_train.append(label)

    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)

    pickle_x_train_out = open("X_train.pickle", "wb")
    pickle.dump(X_train, pickle_x_train_out)
    pickle_x_train_out.close()

    pickle_y_train_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_y_train_out)
    pickle_y_train_out.close()


def create_test_data():
    print("CREATING TEST DATA")
    for category in CATEGORIES_TEST:
        path = os.path.join(DATADIR_TEST,category)
        class_num = CATEGORIES_TRAIN.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])  # add this to our test_data
            except Exception as e:
                pass
            
    X_test = []
    y_test = []

    for features,label in test_data:
        X_test.append(features)
        y_test.append(label)

    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.array(y_test)

    pickle_x_test_out = open("X_test.pickle", "wb")
    pickle.dump(X_test, pickle_x_test_out)
    pickle_x_test_out.close()

    pickle_y_test_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_y_test_out)
    pickle_y_test_out.close()