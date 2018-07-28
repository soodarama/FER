# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 12:16:44 2018

@author: sooda
"""
# Reference
# https://www.safaribooksonline.com/library/view/machine-learning-with/9781787121515/1912b6fc-4360-4836-b658-0f07f39a275c.xhtml
# Has whole code explained - let's try this out

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd

#Dataset consists of gray scale face photos encoded as 
#pixel intensities. 48 x 48 gives 2304 pixels for each. 
#Every image is marked according to the emotion on the face.
#data = pd.read_csv("C:/Users/srama/Documents/Research/ICCST2018/fer2013.tar/fer2013/fer2013.csv") 
data = pd.read_csv("C:/Users/Nallini/Documents/Research/2017 Sepi/Facial Expression with CNN/fer2013/fer2013.csv");
print(data.head() )
print( data.emotion.value_counts() )

#Here 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, and 6=Neutral.
#Let's remove Disgust, as we have too little samples for it:
data = data[data.emotion != 1] 
data.loc[data.emotion > 1, "emotion"] -= 1 
print (data.emotion.value_counts())

emotion_labels = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"] 
num_classes = 6

#This is how samples are distributed among training and test. We'll be using training to train the model and everything else will go to test set:
print( data.Usage.value_counts()  )
# Training =28273, PrivateTest=3534, PublicTest=3533

#The size of images and the number of channels (depth):
from math import sqrt 
depth = 1 
height = int(sqrt(len(data.pixels[0].split()))) 
width = int(height) 
print( height )

#Display some faces
import numpy as np 
import scipy.misc 
from IPython.display import display 
for i in range(0, 5): 
    array = np.mat(data.pixels[i]).reshape(48, 48) 
    image = scipy.misc.toimage(array, cmin=0.0) 
    display(image) 
    print(emotion_labels[data.emotion[i]])
    
#Many faces have ambiguous expressions, so our neural 
#network will have a hard time classifying them. For 
#example, the first face looks surprised or sad, rather 
#than angry, and the second face doesn't look angry at all. 
#Nevertheless, this is the dataset we have. For the real 
#application, I would recommend collecting more samples of 
#higher resolution, and then annotating them such that 
#every photo is annotated several times by different 
#independent annotators. Then, remove all photos that were 
#annotated ambiguously.
    
#..................
#Splitting the data into Training and Test sets
#..................
train_set = data[(data.Usage == 'Training')] 
test_set = data[(data.Usage != 'Training')] 
#X_train = np.array(map(str.split, train_set.pixels), (np.float32)) 
#X_testn = np.array(map(str.split, test_set.pixels), (np.float32)) 
#print( (X_train.shape, X_test.shape))


#Separate train data
train_set = data[data.Usage == "Training"]
Xtrain = train_set.pixels.str.split(" ").tolist() # Note: 'pixels' is an EXCEL heading
Xtrain = pd.DataFrame(Xtrain, dtype=int)
X_train = Xtrain.values


#Separate test data
test_set = data[data.Usage != "Training"]
Xtest = test_set.pixels.str.split(" ").tolist()  # Note: 'pixels' is an EXCEL heading
Xtest = pd.DataFrame(Xtest, dtype=int)
X_test = Xtest.values

(X_train.shape, X_test.shape) # ((28273, 2304), (7067, 2304))
X_train = X_train.reshape(28273, 48, 48, 1) 
X_test = X_test.reshape(7067, 48, 48, 1) 
(X_train.shape, X_test.shape) # ((28273, 48, 48, 1), (7067, 48, 48, 1))

num_train = X_train.shape[0]
num_test = X_test.shape[0]
(num_train, num_test) # (28273, 7067)

#Converting labels to categorical:
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values Using TensorFlow backend. 
y_train = train_set.emotion 
y_train = np_utils.to_categorical(y_train, num_classes) 
y_test = test_set.emotion 
y_test = np_utils.to_categorical(y_test, num_classes) 


#Define a function to show image through 48*48 pixels
def show(img):
    show_image = img.reshape(48,48)
    
    #plt.imshow(show_image, cmap=cm.binary)
    plt.imshow(show_image, cmap='gray')

#show one image
show(images[7])
