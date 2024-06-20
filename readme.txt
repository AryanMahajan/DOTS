This project, DOTS (Dog or Cat Sorter), utilizes TensorFlow and Convolutional Neural Networks (CNNs) to accurately classify images containing either dogs or cats.

Project Description:
DOTS is an image classification project built with TensorFlow. It aims to achieve high accuracy in distinguishing between images of dogs and cats. The model is trained on a large dataset of labeled dog and cat images.

Built With:
TensorFlow: Deep learning framework used for building and training the CNN model.
Keras: High-level API built on top of TensorFlow, simplifying model building.

Requirements for the project are mentioned in the requirements.txt file.

How to run the model:
1. After downloading or cloning the repository. Go to the "data" folder then to "PetImages" then to  "Test" then make ONE folder named "Dog" or "Cat" then paste the photo of a dog or cat respectively.
2. Then open category.py file and change the "TEST_CATEGORY" ot the name of the folder you chose to made in step 1 (Dog or Cat).
3. Now also set the "DATADIR_TRAIN" and "DATADIR_TEST" to the "Location-of-repository-on-your-system\data\PetImages\Train" and "Location-of-repository-on-your-system\data\PetImages\Test" respectively.
4. Then go to the terminal and type "python main.py" to run the model and make a prediction.

Inbuilt model
The model "dots.keras" is already trained on a dataset of 25,00 images (12,500 dogs and 12,500 cats).

Contributor: Aryan Mahajan
GitHub Username: AryanMahajan