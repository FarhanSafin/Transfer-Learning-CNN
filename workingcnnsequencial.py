# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:54:41 2021

@author: Farhan
"""


# import library
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalMaxPooling2D
from keras.optimizers import RMSprop
from scipy.io import loadmat
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix


# function 
# Save plot
import matplotlib.pyplot as plt
def plotsave(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validaton'], loc='upper left')
    plt.savefig('histryofloss.pdf', bbox_inches='tight')
    pass


# Load Data
data = loadmat('usagedata2D.mat') # loading the data into dictionary format
locals().update(data)


# re defined # from dictionary
set1= L0_2
set2= L1_2
set3= L2_2
U_L = Real_label
T_L = Real_label


# Normalize the data
set1 = set1 / 255.0
set2 = set2 / 255.0
set3 = set3 / 255.0

# different types of normalization we can do
# if image << 255.0 
# max normalization
# min normalization
# standardization [upb lwb] 



# 3D tensor 
# neural network input is always a tensor
# higher dimension matirx
set1 = np.reshape(set1,(len(set1),32,len(np.transpose(set1)),1))
set2 = np.reshape(set2,(len(set2),32,len(np.transpose(set2)),1))
set3 = np.reshape(set3,(len(set3),32,len(np.transpose(set3)),1))



# Encoding Data Label 
# -- 1. binarization 
# -- 2. One hot encoding
 
label_encoder  = LabelBinarizer()
U_L  = label_encoder.fit_transform(U_L)
T_L  = label_encoder.fit_transform(T_L)

# task finished
# 1. Data read
# 2. data Normalize
# 3. label binarize

# Set the random seed
random_seed = 2

'DATA SPLITING'
# " Change Here"
# Exp1
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(set3, T_L, test_size = 0.3, random_state=random_seed)

# Set the CNN model 
# assistive link: https://keras.io/examples/mnist_cnn/
#https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/?fbclid=IwAR03s_rWD9YPXUfjuvFLkN3pxwdRuwOvUt7sYK5-Vv9hXk-YMEw-e_g41eE

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (32,32,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# https://medium.com/@iamvarman/how-to-calculate-the-number-of-parameters-in-the-cnn-5bd55364d7ca


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
keras.layers.GlobalMaxPooling2D(data_format='channels_last')
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))

# output layer
model.add(Dense(4, activation = "softmax")) # output softmax
# Define the optimizer
# https://keras.io/ko/optimizers/
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) # <<
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 2 #<<100
batch_size = 32 # <<

# intuively epoch means first we divide the data into epoch time
# batch is intuitively divide each epoch into batch time
# show model
model.summary()

# Without data augmentation
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val), verbose = 1)

plotsave(history)

  
#3 . set 1 set2 set3
# training phase : set 3 -- 70 % train and 30% validation
# actual testing phase: set1


##################################### TEST 1  
# Predict the values from the  dataset
Y_pred = model.predict(set1)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(U_L,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
print(confusion_mtx)
target_names = ['normal', 'ball', 'inner','outer6']
# Argmax function is deconverting from hot encoding to simple one
print (classification_report(Y_pred_classes,Y_true, target_names = target_names))

model.save("complete_saved_model/")

