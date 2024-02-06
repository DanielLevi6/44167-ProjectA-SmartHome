# https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy/notebook

import pandas as pd
import cv2
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
"""
Labels-
- A(0)- airconditioner
- B(1)
- C(2)
- F(5)
- L(11)- light
- T(19)- television
- U(20)
- V(21)
- W(22)
- Y(24)
"""

# Data load
train_df = pd.read_csv("../data/train/sign_mnist_train.csv")
test_df = pd.read_csv("../data/test/sign_mnist_test.csv")

# Data pre-processing
train_labels = train_df['label']
test_labels = test_df['label']

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values

# We perform a grayscale normalization to reduce the effect of illumination's differences.
# Moreover the CNN converges faster on [0..1] data than on [0..255].

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255

# Add direction redundancy
iterations = len(x_train)

# Reshaping the data from 1-D to 3-D as required through input by CNN's HWC
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
# x_train = [sample.resize(64,64) for sample in x_train]
# x_test = x_test.resize(64,64)

for i in range(iterations):
        x_horizonal_flip = cv2.flip(x_train[i], 1)
        x_horizonal_flip = x_horizonal_flip.reshape(1, 28, 28, 1)
        x_train = np.concatenate((x_train, x_horizonal_flip), 0)
        y_train = np.concatenate((y_train, y_train[i].reshape(1, 24)))
        print(f"Image {i} has been flipped")

# # Preview of first 10 images
# f, ax = plt.subplots(2,5)
# f.set_size_inches(10, 10)
# k = 0
# for i in range(2):
#     for j in range(5):
#         ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
#         k += 1
#     plt.tight_layout()

"""
Data Augmentation
In order to avoid overfitting problem, we need to expand artificially our dataset. 
We can make your existing dataset even larger. The idea is to alter the training data with small transformations to 
reproduce the variations.

Approaches that alter the training data in ways that change the array representation while keeping the label the same 
are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, 
vertical flips, random crops, color jitters, translations, rotations, and much more.

By applying just a couple of these transformations to our training data, we can easily double or triple the number of 
training examples and create a very robust model.
"""
# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

"""
For the data augmentation, i choosed to :

Randomly rotate some training images by 10 degrees Randomly Zoom by 10% some training images Randomly shift images 
horizontally by 10% of the width Randomly shift images vertically by 10% of the height I did not apply a vertical_flip 
nor horizontal_flip since it could have lead to misclassify.

Once our model is ready, we fit the training dataset .
"""

# Training The Model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.0001)

# Model
from resnet18 import ResNet18
model = ResNet18(24)
model.build(input_shape = (None,28,28,1))
model.compile(optimizer = "adam",loss='categorical_crossentropy', metrics=["accuracy"])

from keras.callbacks import EarlyStopping

es = EarlyStopping(patience= 8, restore_best_weights=True, monitor="val_acc")
#I did not use cross validation, so the validate performance is not accurate.
STEPS = len(x_train) / 256
history = model.fit(datagen.flow(x_train,y_train,batch_size = 32), steps_per_epoch=STEPS, epochs=100, validation_data=(x_train, y_train),callbacks=[es])

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

model.save('../model/resnet18')

# model = Sequential()
# model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,1)))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Flatten())
# model.add(Dense(units = 512 , activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(units = 24 , activation = 'softmax'))
# model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
# model.summary()

# history = model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 20 , validation_data = (x_train, y_train) , callbacks = [learning_rate_reduction])

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

model.save('../model/resnet18')
