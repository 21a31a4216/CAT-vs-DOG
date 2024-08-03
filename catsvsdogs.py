# -*- coding: utf-8 -*-
"""CatsVsDogs (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Oka1UNRDe6kFhzDJNmZvtaFXIOkA8mpR
"""

# Commented out IPython magic to ensure Python compatibility.
#Importing necessary libraries
import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

"""## SAMPLE IMAGES OF THE DATASET

![image.png](attachment:image.png)

## **Dataset containing 4000 pictures of cats and dogs (2000 cats, 2000 dogs). We will use 2000 pictures for training, 1000 for validation, and finally 1000 for testing**

![image.png](attachment:image.png)
"""

train_dir = r'..\data\train'  # Location of training images
validation_dir = r'..\data\validation' #Location of validation images
test_dir =r'..\data\test' #Location of test images

"""#

# I. MY MODEL

### Data Preprocessing

* Read the picture files.
* Decode the JPEG content to RBG grids of pixels.
* Convert these into floating point tensors.
* Rescale the pixel values (between 0 and 255) to the [0, 1] interval
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""Link : https://keras.io/preprocessing/image/

**We will be feeding the data into the model using ImageDataGenerator using flow_from1_directory method which will create an array iterator, this iterator will pick up images in specified batches from the specified directory and automatically feed it to the network for training.**
"""

# Generating batches of tensor image data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

"""**Although the Iterator will feed your data into the model and the labels will be assigned according to the file names, eg: my cat images are in cat folder so they will be assigned a label cat and similarly for dogs, obviously they would be coded to 0&1 before feeding into the model, but all this is done by the Iterator, so how will we know what are the labels??**

**Hence I have written a simple function which will plot a batch of images from the Iterator along with the assigned label, this is done using next() function which yeilds a batch of images from the Iterator and the class they belong to**
"""

def plots(ims, figsize=(16,16), rows=4, interp=False, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])

imgs, labels = next(train_generator)
plots(imgs, titles=labels)

"""## As we can see cats are label 0 whereas dogs are label 1"""

class_names = ['Cat','Dog'] #Creating a dictionary of class names according to the label

"""### MODEL ARCHITECTURE"""

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

#Plotting a graphical representation of the model
import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot

keras.utils.plot_model(model)

model.summary()

from tensorflow.keras import optimizers

model.compile(loss='binary_crossentropy',                 #Since we have two classes I am using a binary cross entropy
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

"""* **For calculation of steps per epoch :**
  **Our train_generator is feeding data in batchs of 20 and the size of out test set is 2000
  hence maximum steps_per_epoch = 2000/20 = 100, for this value the whole data set will be feeded.**
* **Similarly for validation steps, our validation set is of size 1000 and we have a batch size of 20,
  thus maximum validation_steps = 50**

## TRAINING

**If you do not have a dedicated GPU then remove the tf.device line**
"""

with tf.device("/device:GPU:0"):      #To make sure that  the GPU is being used for training
    history = model.fit_generator(    #Feeding the data from iterator into the model using fit_generator() method
      train_generator,
      steps_per_epoch=100,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50)

"""## After training:
* **Validation accuracy:72.8%**
* **Training accuracy:92.55%**
"""

pd.DataFrame(history.history).plot(figsize=(8, 5))  #Plotting the model history by converting the data of epochs into a DataFrame
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

"""As we can see the model overfits and is not converging, maybe if we train it for some more epochs we can acheive a validation accuracy of 80%"""

model.save("my_model.h5")  #Saving the model weights to avoid retraining everytime

"""## LOADING THE MODEL USING THE .h5 file"""

model = tf.keras.models.load_model('my_model.h5')
model.summary()

"""## TESTING

### **Now we will test our model on the test set**
"""

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

"""**For plotting the results I have written this code, first of all using the next() method I load a batch of 20 images into the imgs variable.**

**Now for predicting and plotting the images one by one we need to pass the images with a batch size of 1, if you will print the shape of imgs the result is (20,150,150,3), hence if we pass this to the predictor then instead of getting a single prediction we will get an array of 20 oredicted values which then we will have to match one by one to the image.**

**Hence I first take every image from imgs and convert it to a batch size of 1 i.e the shape would now be (1,150,150,3). This is explained better below when I test a random image using the model**

**Now I pass this image to the model and the predicted value is compared with a threshold of 0.5 more like rounding off.**

**Since Dog is class 1 hence if predicted value is greater than 0.5 we predict it as a Dog and print the confidnece that is the predicted value.**

**Cat is class 0 hence if predicted value is less than 0.5 we predict it as a Dog and print the confidnece that is the (1-predicted value).**


"""

imgs, labels = next(test_generator)
fig=plt.figure(figsize=(16, 16))
columns = 4
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img_t= np.expand_dims(imgs[i], axis=0)
    prediction = model.predict(img_t)
    if(prediction[:,:]>0.5):
        value ='Dog :%.2f'%(prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    else:
        value ='Cat :%.2f'%(1-prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    plt.imshow(imgs[i])

model.evaluate_generator(test_generator, steps=50)

"""### **As we can see we get a testing accuracy of 72.29%**

### Now we will test our model on random cat and dog images from the internet

Knowing how to preprocess images and use them for testing is an important and crucial step, due to increased reasearch nowdays we get prepared and cleaned datasets, however it is necessary to know how to process your own data, rather than just loading it using one command

**There are many methods in which you can view your images in python:**
* Open CV
* keras.preprocessing
* Pillow Library
* many more... (which I have not discovered yet)

**I am testing my model on two random images from the net, I will be showing ways to preprocess images using keras and pillow library, openCV is also a popular and very efficient method but the other two methdos are also pretty good and can perform simple operations on images, while OpenCV is dedicated for Vision and advanced processing of images, you can easliy get the implementation using OpenCV on the net, but I have not it in this project.**

**We are using binary cross entropy as the loss, our class0 is Cat and class1 is Dog. The output if the function will be between 0 &1.**
* If the output of the model is closer to 0 it is a Cat
* If the output of the model is closer to 1 it is a Dog

## Using Keras preprocessing
"""

from keras.preprocessing import image

#Loading and plotting the image
img = image.load_img('confuse.jpg', target_size=(150, 150))   #resizing and loading the image load_img method of keras
plt.imshow(img)        #imshow is a method of matplotlib for viewing images

"""As seen above this is an image containing many dogs and cats, the dogs are in the front and bigger in size and hence I expect the model to predict this image belonging to class Dog with a high confidence, also given that the accuracy was just 72%."""

test_image = image.img_to_array(img)  #Converting the jpg file to array of pixels
test_image.shape   #Checking the shape of image

"""**Althoguh this testing dimensions match with our input size, but if you directly feed the image to the model it will give you an error of bad dimensions, because whenever we feed an image to a model it is feeded in batches, the first dimension is taken as the batch size and rest as the input shape,in prepared datasets you get all this processsing done already hence testing on your ownm data is very important! When the Iterator was feeding the image, the batch size was 16 but here we are giving a single image hence the batch size is 1 so we will add a new axis '0' to the image using expand_dims, we can also resize the image, I have done that in the next example.**"""

test_image = np.expand_dims(test_image, axis=0)
test_image.shape

"""**bbox is used to display text on image, read the matplotlib documentation for more info.**"""

pred=model.predict(test_image)             #Predicting the output
print("Predicted value: {}".format(pred))
pred=pred.round()                          #Rounding off the output to nearest class 0 or 1
plt.title('Test Image')
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  #getting the class name from dictionary
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) #Displaying the class along with class value
plt.imshow(img)

"""**As we can see this model predicts this image as a dog with 100% confidence, as the label for Dog is 1 and the predicted value id also 1!! Majorly because the accuracy was moderate and the model was overfitting, so it doesn't identify the cats in the given picture at all!**

## USING PILLOW(Python Imaging Library)
"""

from PIL import Image
img=Image.open('cat.jpg')     #The open method will just identify the image but the file image data is not read,indeed a lazy function!
print(img)
plt.title('Original Image')
plt.imshow(img)               #Hence we will plot the image using imshow method from matplotlib

"""As seen above this is an image of a cat wearing a mask! I expect the model to predict it as a cat with high confidence as the image contains just one object.

**PIL library has many more fucntions like rotate, alpha, blend etc for performing operations on image.**
"""

new_img=img.resize((150,150)) #Resizing the image using resize() method of PIL library
new_img.save('cat_new.jpg')   #Saving the resized image
plt.title('Test Image')
plt.imshow(new_img)

"""As you can see in above two cells I have used plt.imshow() to plot the images, I have merely passed PIL Image object to it. The imshow() method can take an array of pixels or a PIL Image as input. Previously we had used expand dims to add the new axis representing the batch size, here I will be implementing that using array indexing but for that I need to convert my Image into an array, for that I will be using the imread() method."""

new_img = plt.imread('cat_new.jpg')
new_img=new_img/255.
new_img=new_img[np.newaxis,:,:,:]
new_img.shape

pred=model.predict(new_img)              #Predicting the output
print("Predicted value: {}".format(pred))
pred=pred.round()                        #Rounding off the output to nearest class 0 or 1
plt.title('Original Image')
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  #getting the class name from dictionary
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) #Displaying the class along with class value
plt.imshow(img)                                                                            #Read the matplotlib documentation for more info.

"""**As we can see this model predicts this image as a cat with 97% confidence, as the label for Cat is 0 and the predicted value is 0.03. This is a pretty good result but we should train our model to avoid that 3% error as well. Ideally the output should be 0 as Cat belongs to class 0.**

**Basically the model predcits the image as 97% Cat and 3 % Dog!**
"""

from tensorflow.keras import backend as K

K.clear_session() # Clearing the variables of previous session
del model   #Deleting the previous model so that we can load our new model and perform test on it

"""#

**======================================================================================================================**
**======================================================================================================================**

#

# II. DATA AUGMENTATION

## The testing and validation is performed for the same images in all models hence for detailed explanation refer the testing and preprocessing parts of MY MODEL

Shear & Rotation:
![image.png](attachment:image.png)

The textbook definition of data augmentataion would be:

Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks.

Here I would be using the ImageDataGenrator from keras to perform various operations on my training images like shear, rescale, rotate etc. For more information visit the Keras official documentation. The need to do augmentation is because I only have 2000 training images which is a small number to be honest; given the amount of data being generated regualarly, so I would be performing various operations on my data and then feeding it to the model, now my model is trained to identify a rotated cat as a cat and not some othe object, thereby increasing the performance of my model!!

**Note: The method which I have used just performs data augmentation and feeds the images to the model, for every batch it performs the given operations randomly on the images and feeds the batch of images to the model, overall 2000 images are only being fed, no new data is genrated. For saving the augmented images you have to use the save_to_dir attribute and then add those data to your train folder so now you will have more images! For more information visit the Keras official documentation.**
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary')

"""### Note that I have performed the data augmentation only on the training set and not on the test and validation set because those images are used for testing and not training the model so we don't need to process them."""

class_names = ['Cat','Dog']

"""## Sample of how the images would look after shear, shift, rotate etc..

![image.png](attachment:image.png)
"""

from tensorflow.keras import optimizers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))                    # Adding a dropout layers since training is performed for 100 epochs,
model.add(layers.Dense(512, activation='relu'))   # and this would reduce the training time.
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

#Plotting a graphical representation of the model
import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot

keras.utils.plot_model(model)

"""## TRAINING"""

with tf.device("/device:GPU:0"):
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

"""## After training:
* **Validation accuracy: 81.25%**
* **Training accuracy: 82.50%**
"""

model.save("augmented_cnn.h5")

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

"""As we can see this model converges better than last one however it still doesn't converge properly and tends to overfit!! Also the loss is a bit high...

One of the reason of lower performance is that I am not utilizing all the training data, as you can see the batch size was 16 and I could have had 125 maximum steps per epoch, that would have got me 2-4% higher accuracy.

## LOADING THE MODEL USING THE .h5 file
### MODEL ARCHITECTURE
"""

model = tf.keras.models.load_model('augmented_cnn.h5')
model.summary()

"""## TESTING
### The proper explanation of testing code is given in the TESTING section of MY MODEL. The same images are used for testing all the models.
"""

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

imgs, labels = next(test_generator)
fig=plt.figure(figsize=(16, 16))
columns = 4
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img_t= np.expand_dims(imgs[i], axis=0)
    prediction = model.predict(img_t)
    if(prediction[:,:]>0.5):
        value ='Dog :%.2f'%(prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    else:
        value ='Cat :%.2f'%(1-prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    plt.imshow(imgs[i])

model.evaluate_generator(test_generator, steps=50)

"""### **As we can see we get a testing accuracy of 80.9%**"""

from PIL import Image
img=Image.open('cat.jpg')
new_img=img.resize((150,150))
new_img.save('cat_new.jpg')

new_img=plt.imread('cat_new.jpg')
new_img.shape
plt.title('Test image')
plt.imshow(new_img)

new_img=new_img/255.
new_img=new_img[np.newaxis,:,:,:]
new_img.shape

pred=model.predict(new_img)                  #Predicting the output
print("Predicted value: {}".format(pred))
pred=pred.round()                            #Rounding off the output to nearest class 0 or 1
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  #getting the class name from dictionary
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) #Displaying the class along with class value
plt.imshow(img)

"""**As we can see this model predicts this image as a cat with 84.49% confidence, as the label for Cat is 0 and the predicted value is 0.155. This means this model performs worser than the previous one for this particular image, where we had got a 97% confidence!! Hence even if a model has high accuracy, it doesn't mean it will perform well on unseen data!**

**Basically the model predcits the image as 84.49% Cat and 15.5% Dog!**
"""

from PIL import Image
img=Image.open('confuse.jpg')
new_img=img.resize((150,150))
new_img.save('confuse_new.jpg')

new_img=plt.imread('confuse_new.jpg')
plt.title('Test image')
plt.imshow(new_img)

new_img=new_img/255.
new_img=new_img[np.newaxis,:,:,:]
new_img.shape

pred=model.predict(new_img)                    #Predicting the output
print("Predicted value: {}".format(pred))
pred=pred.round()                              #Rounding off the output to nearest class 0 or 1
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  #getting the class name from dictionary
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) #Displaying the class along with class value
plt.imshow(img)

"""**As we can see this model predicts this image as a dog with 97.34% confidence, as the label for Dog is 1 and the predicted value is also 0.9734. Unlike last image the model performs better on this image than the previous one, since now it also is taking in account the cats present in the pictures. The previous model had predicted this image as a Dog with 100% confidence!**

**Although talking statistically this image has 9 cats and 8 dogs hence statistically the picture has around 53% cats and 47% dogs!! Hoewever visually we can see that the dogs are bigger in size and take up more pixels than the cats and probably that is why our model predicts the image as a Dog with high confidence, the model is trained just to classify the image as cat or a dog, this is not an obeject detection model so the classification will not be based on the count of dogs and cats!!!**

**Basically the model predcits the image as 97.34% Dog and 2.66% Cat!**
"""

from tensorflow.keras import backend as K

K.clear_session() # Clearing the variables of previous session
del model   #Deleting the previous model so that we can load our new model and perform test on it

"""#

**======================================================================================================================**
**======================================================================================================================**

#

# III. TRANSFER LEARNING USING VGG16
## The testing and validation is performed for the same images in all models hence for detailed explanation refer the testing and preprocessing parts of MY MODEL

### PERFORMING DATA AUGMENTATION
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

class_names = ['Cat','Dog'] #Creating a dictionary of class names according to the label

"""## IMPORTING THE VGG 16 MODEL WEIGHTS for IMAGENET DATASET"""

from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

"""## VGG16 ARCHITECTURE"""

from IPython.core.display import Image, display
display(Image('https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png', width=700, unconfined=True))

conv_base.summary()

"""## ADDING LAYERS TO THE VGG 16 Architecture"""

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#Plotting a graphical representation of the model
import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot

keras.utils.plot_model(model)

model.summary()

"""**VGG 16 has a heavy architecture and hence if you do not have a dedicated GPU it may take 8-10 hours to training the whole network with the added layers, so you can freeze the VGG16 layers and use the pretrained weights and just train your additional layers; to do so uncomment the next code block**"""

# conv_base.trainable = False

"""I am using RMSprop as an optimizer with a very small learning rate for better results, you can try with Adam optimizer, but Adam optimizer works for some predefined values of learning rate, changing them may not give you a satisfactory result or maybe it will!! Hyperparmeter tuning is still a big challange!"""

from tensorflow.keras import optimizers
with tf.device("/device:GPU:0"):
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

"""**Since the training process will take well enough time hence it is safe to create checkpoints after every epoch!
Also you can resume the training process from where we left off in case it was interrupted or for fine-tuning the model**
"""

checkpoint_cb = keras.callbacks.ModelCheckpoint("CNN_Project_Model-{epoch:02d}.h5")

"""### TRAINING"""

with tf.device("/device:GPU:0"):
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=[checkpoint_cb])

"""# After training:
* **Validation accuracy:96.1%**
* **Training accuracy:98.65%**
"""

model.save("vgg16_cnn.h5")   #Saving the model to avoid retraining

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

"""As we can see this model converges pretty well!! But the spike at the 23rd epoch is very peculiar, as we can see at the 23rd epoch the loss increases suddenly and thereafter things are back on track.

The reason to this is something very intrinsic to the way optimizers work, note that I had used RMSprop as an optimizer, if you go through the algorithm you will get to know that it uses mini batches of the data for optimizing weights, other optimizers like Stochastic Gradient Descent(batch=1), Adam etc also use mini batches.

When mini batches are used to update weights the spikes are an unavoidable consequence. Some mini-batches have 'by chance' unlucky data for the optimization!! The same doesn't happen in (Full) Batch Gradient Descent because it uses all training data in each optimization epoch.

### Saving the model history at every epoch into a DataFrame as I don't wanna loose this data, even after using a dedicated GPU it took me an average of 90 seconds per epoch to train the complete model!
"""

hist_df = pd.DataFrame(history.history)

hist_df.head()

hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

"""## LOADING THE MODEL USING THE .h5 file
### MODEL ARCHITECTURE
"""

model = tf.keras.models.load_model('vgg16_cnn.h5')
model.summary()

"""## TESTING
#### The proper explanation of testing code is given in the TESTING section of MY MODEL. The same images are used for testing all the models.
"""

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

imgs, labels = next(test_generator)
fig=plt.figure(figsize=(16, 16))
columns = 4
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img_t= np.expand_dims(imgs[i], axis=0)
    prediction = model.predict(img_t)
    if(prediction[:,:]>0.5):
        value ='Dog :%.2f'%(prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    else:
        value ='Cat :%.2f'%(1-prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    plt.imshow(imgs[i])

"""### As we can see compared to last two models this model gives the best and appropriate predictions"""

model.evaluate_generator(test_generator, steps=50)

"""### **As we can see we get a testing accuracy of 95.8%**"""

from PIL import Image
img=Image.open('cat.jpg')
new_img=img.resize((150,150))
new_img.save('cat_new.jpg')

img_test=plt.imread('cat_new.jpg')
img_test.shape
plt.title('Test Image')
plt.imshow(img_test)

img_test=img_test/255.
img_test=img_test[np.newaxis,:,:,:]
img_test.shape

pred=model.predict(img_test)                     #Predicting the output
print('Output of Model:{}'.format(pred))
pred=pred.round()                                #Rounding off the output to nearest class 0 or 1
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  #getting the class name from dictionary
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) #Displaying the class along with class value
plt.imshow(img)

"""**As we can see this model predicts this image as a cat with almost 100% confidence, as the label for Cat is 0 and the predicted value is 2e-15 which is as good as zero. This means this model perfectly classifies this image as a cat unlike previous two!**"""

from PIL import Image
img=Image.open('confuse.jpg')
new_img=img.resize((150,150))
new_img.save('confuse_new.jpg')

new_img=plt.imread('confuse_new.jpg')
plt.title('Test image')
plt.imshow(new_img)

new_img=new_img/255.
new_img=new_img[np.newaxis,:,:,:]
new_img.shape

pred=model.predict(new_img)                    #Predicting the output
print("Predicted value: {}".format(pred))
pred=pred.round()                              #Rounding off the output to nearest class 0 or 1
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  #getting the class name from dictionary
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) #Displaying the class along with class value
plt.imshow(img)

"""**As we can see this model predicts this image as a dog with 77.58% confidence, as the label for Dog is 1 and the predicted value is also 0.7758. As discussed in the previous model the picture has around 53% cats and 47% dogs "statistically", and in comparison to previous model the confidence has come doen to 77% from 97 % hence this model definitely performs better than the previous one on this image**

**Basically the model predcits the image as 77.58% Dog and 22.42% Cat!**

**Maybe making the model more complex will increase the performance, I doubt not as the model history shows that the model is converging pretty well and usually deeper models tend to overfit, VGG16 itself is quite a deep architecture. But when model start to get deep things are pretty uncertain in between, there is no harm in testing until you have enough compuatation power!**
"""

from tensorflow.keras import backend as K

K.clear_session() # Clearing the variables of previous session
del model   #Deleting the previous model so that we can load our new model and perform test on it

"""#

**----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

#

CNNs derive their name from the “convolution” operator. The primary purpose of Convolution in case of a CNNs is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data.

Let us look at the outputs of the of the convolutional layers for an image! In the forst model there were total of 4 convolutional layers with 32,64,128,128 filters respectively. The output of these convolutional layers is the combined output of all these filters!!

### Let us first load our model!
"""

model = tf.keras.models.load_model('my_model.h5')
model.summary()

"""### Loading and processing our test image"""

from keras.preprocessing import image
img = image.load_img('cat.jpg', target_size=(150, 150))
plt.imshow(img)
test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image, axis=0)

"""### Let us look at the output of individual convolutional layers for this image!!

"""

#Printing the name of layers in the model
model_layers = [ layer.name for layer in model.layers]
print('layer name : ',model_layers)

#Storing the output of all the four convolutional layers
from tensorflow.keras.models import Model
conv2d_output = Model(inputs=model.input, outputs=model.get_layer('conv2d').output)
conv2d_1_output = Model(inputs=model.input,outputs=model.get_layer('conv2d_1').output)
conv2d_2_output = Model(inputs=model.input,outputs=model.get_layer('conv2d_2').output)
conv2d_3_output = Model(inputs=model.input,outputs=model.get_layer('conv2d_3').output)

#Predicting the output of these layers for our test image
conv2d_features = conv2d_output.predict(test_image)
conv2d_1_features = conv2d_1_output.predict(test_image)
conv2d_2_features = conv2d_2_output.predict(test_image)
conv2d_3_features = conv2d_3_output.predict(test_image)
print('First conv layer feature output shape : ',conv2d_features.shape)
print('Second conv layer feature output shape : ',conv2d_1_features.shape)
print('Third conv layer feature output shape : ',conv2d_2_features.shape)
print('Fourth conv layer feature output shape : ',conv2d_3_features.shape)

"""## Output of first convolutional layer"""

fig=plt.figure(figsize=(15,8))
columns = 8
rows = 4
for i in range(columns*rows):
    #img = mpimg.imread()
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_features[0, :, :, i], cmap='gray')
plt.show()

"""## Output of second convolutional layer"""

fig=plt.figure(figsize=(15,9))
columns = 8
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_1_features[0, :, :, i], cmap='gray')
plt.show()

# Note that we are looking at the output for first 32 filters only, this layer has 64 filters,
# if you wish to see the output for a particular filter then uncomment the code below and replace
# i with the filter whose output you want to see!!

# plt.imshow(conv2d_1_features[0, :, :, i], cmap='gray')

"""
## Output of third convolutional layer"""

fig=plt.figure(figsize=(15,9))
columns = 8
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_2_features[0, :, :, i], cmap='gray')
plt.show()

# Note that we are looking at the output for first 32 filters only, this layer has 128 filters,
# if you wish to see the output for a particular filter then uncomment the code below and replace
# i with the filter whose output you want to see!!

# plt.imshow(conv2d_2_features[0, :, :, i], cmap='gray')

"""## Output of fourth convolutional layer"""

fig=plt.figure(figsize=(15,8))
columns = 8
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_3_features[0, :, :, i], cmap='gray')
plt.show()

# Note that we are looking at the output for first 32 filters only, this layer has 128 filters,
# if you wish to see the output for a particular filter then uncomment the code below and replace
# i with the filter whose output you want to see!!

# plt.imshow(conv2d_3_features[0, :, :, i], cmap='gray')

"""These outputs of filters are also known as feature maps. A filter slides over the input image (convolution operation) to produce a feature map. The convolution of another filter, over the same image gives a different feature as observed.

CNN learns the values of these filters on its own during the training process (although we still need to specify parameters such as number of filters, filter size, architecture of the network etc. before the training process). The more number of filters we have, the more image features get extracted and the better our network becomes at recognizing patterns in unseen images!!

We can see that as we go for deeper convolutional layers, the filters extract more specific and intricate features!!!
While the outputs of the first two convolutional layers detect mainly the edges, the third and fourth convolutional layers extract very specific features at the pixel level!!

**However after every convolutional layer I have also added a maxpooling layer, let us look the output of maxpool layer!!**

I will be looking at the output of the first maxpool layer for a particular filter as I just want to verify the operation of maxpooling layer, which is to reduce the spatial size of the feature map hence the resolution of our feature map should be reduced after applying the maxpool layer, for checking that I will also plot the input given to the maxpool layer which is nothing but the output of the preceeding convolutional layer!!
"""

maxpool_output = Model(inputs=model.input,outputs=model.get_layer('max_pooling2d').output)
maxpool_features = maxpool_1_output.predict(test_image)

fig=plt.figure(figsize=(15,8))
fig.add_subplot(1, 2, 1)
plt.title('Filter output')
plt.imshow(conv2d_features[0, :, :, 28], cmap='gray')
fig.add_subplot(1, 2, 2)
plt.title('Maxpool layer output')
plt.imshow(maxpool_features[0, :, :, 28], cmap='gray')
plt.show()

"""As we can observe after applying the maxpooling layer the resolution of our feature map has reduced but the output feature maps still preserves the important features from the input feature map hence the feature map now represents the important features only and is more managable, thereby reducing the computations!!!"""

from tensorflow.keras import backend as K

K.clear_session()
del model

"""## Conclusion
This project is inspired by the Kaggle Challange held in 2003 in which thousands of images of cats and dogs were given
and a model was to be built to classify those images into cats and dogs.

The best accuracy achieved in that competition was 98%!!

I used a subset of that data and built my model, in the original dataset there were around 25000 images for training but I am only using 2000 images..

I used three different achitectures to train this dataset and increased the validation accuracy from around 73% to 96%!!!

My first model was a simple CNN network with four convolutional layers, in my second model I used data augmentation and finally in my final model I used transfer learning using VGG16 and acheived a validation accuracy of 96.1% which is pretty good!!
It is possible to Achieve more accuracy on this dataset using deeper network and fine tuning of network parameters for training. You can download the models from this repository and play with it.
"""