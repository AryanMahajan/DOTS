from training_model import training_model
from loading_data import create_test_data, create_training_data
from category import TEST_CATEGORY as class_category

import tensorflow as tf
import os
import pickle

#Checking if the model exists or not
if os.path.exists("dots.keras"):
    print("Model exists")
    model = tf.keras.models.load_model("dots.keras")
else:
    print("Training Model")
    training_model()
    model = tf.keras.models.load_model("dots.keras")

#Checking if the data is saved or not
if os.path.exists("X_test.pickle") and os.path.exists("y_test.pickle"):
    print("LOADING TEST DATA")
    x_test = pickle.load(open("X_test.pickle", "rb"))
    y_test = pickle.load(open("y_test.pickle", "rb"))
else:
    create_test_data()
    print("LOADING TEST DATA")
    x_test = pickle.load(open("X_test.pickle", "rb"))
    y_test = pickle.load(open("y_test.pickle", "rb"))

#normalizing x_test
x_test = x_test / 255.0

prediction_num = model.predict(x_test)

#converting the prediction to class labels
if((prediction_num)<=0.5):
    print(f"Probability of the being Dog = {(1-prediction_num)*100} %")
    if class_category == ["Cat"]:
        print("The Prediction is Incorrect.")
    else:
        print("The Prediction is Correct.")
else:
    print(f"Probability of the being Cat = {(prediction_num)*100} %")
    if class_category == ["Dog"]:
        print("The Prediction is Incorrect.")
    else:
        print("The Prediction is Correct.")

print("DELETING TEST DATA")
os.remove("X_test.pickle")
os.remove("y_test.pickle")