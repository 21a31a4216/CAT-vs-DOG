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

train_dir = r'..\data\train'  # Location of training images
validation_dir = r'..\data\validation' #Location of validation images
test_dir =r'..\data\test' #Location of test images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

class_names = ['Cat','Dog'] #Creating a dictionary of class names according to the label

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

import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot

keras.utils.plot_model(model)

model.summary()

from tensorflow.keras import optimizers

model.compile(loss='binary_crossentropy',                 
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

with tf.device("/device:GPU:0"):     
    history = model.fit_generator(    
      train_generator,
      steps_per_epoch=100,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50)

pd.DataFrame(history.history).plot(figsize=(8, 5))  
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.save("my_model.h5")  

model = tf.keras.models.load_model('my_model.h5')
model.summary()

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

from keras.preprocessing import image

img = image.load_img('confuse.jpg', target_size=(150, 150))   #resizing and loading the image load_img method of keras
plt.imshow(img)        #imshow is a method of matplotlib for viewing images

test_image = image.img_to_array(img)  
test_image.shape   

test_image = np.expand_dims(test_image, axis=0)
test_image.shape

pred=model.predict(test_image)             
print("Predicted value: {}".format(pred))
pred=pred.round()                          
plt.title('Test Image')
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) 
plt.imshow(img)

from PIL import Image
img=Image.open('cat.jpg')     
print(img)
plt.title('Original Image')
plt.imshow(img)              

new_img=img.resize((150,150)) 
new_img.save('cat_new.jpg')   
plt.title('Test Image')
plt.imshow(new_img)

new_img = plt.imread('cat_new.jpg')
new_img=new_img/255.
new_img=new_img[np.newaxis,:,:,:]
new_img.shape

pred=model.predict(new_img)              
print("Predicted value: {}".format(pred))
pred=pred.round()                        
plt.title('Original Image')
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) 
plt.imshow(img)                                                                            

from tensorflow.keras import backend as K

K.clear_session() 
del model   

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

class_names = ['Cat','Dog']

from tensorflow.keras import optimizers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))                    
model.add(layers.Dense(512, activation='relu'))   
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot

keras.utils.plot_model(model)

with tf.device("/device:GPU:0"):
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

model.save("augmented_cnn.h5")

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model = tf.keras.models.load_model('augmented_cnn.h5')
model.summary()

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

pred=model.predict(new_img)                  
print("Predicted value: {}".format(pred))
pred=pred.round()                            
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) 
plt.imshow(img)

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

pred=model.predict(new_img)                    
print("Predicted value: {}".format(pred))
pred=pred.round()                              
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) 
plt.imshow(img)

from tensorflow.keras import backend as K

K.clear_session() 
del model   

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

class_names = ['Cat','Dog'] 

from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

from IPython.core.display import Image, display
display(Image('https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png', width=700, unconfined=True))

conv_base.summary()

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot

keras.utils.plot_model(model)

model.summary()

from tensorflow.keras import optimizers
with tf.device("/device:GPU:0"):
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("CNN_Project_Model-{epoch:02d}.h5")

with tf.device("/device:GPU:0"):
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=[checkpoint_cb])

model.save("vgg16_cnn.h5")   

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

hist_df = pd.DataFrame(history.history)

hist_df.head()

hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model = tf.keras.models.load_model('vgg16_cnn.h5')
model.summary()

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

pred=model.predict(img_test)                     
print('Output of Model:{}'.format(pred))
pred=pred.round()                                
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) 
plt.imshow(img)

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

pred=model.predict(new_img)                   
print("Predicted value: {}".format(pred))
pred=pred.round()                              
value ='{} :{}'.format(class_names[int(pred)],pred[0,0])  
plt.text(20, 58,value,color='red',fontsize=15,bbox=dict(facecolor='white',alpha=0.8)) 
plt.imshow(img)

from tensorflow.keras import backend as K

K.clear_session() 
del model   

model = tf.keras.models.load_model('my_model.h5')
model.summary()

from keras.preprocessing import image
img = image.load_img('cat.jpg', target_size=(150, 150))
plt.imshow(img)
test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image, axis=0)

model_layers = [ layer.name for layer in model.layers]
print('layer name : ',model_layers)

from tensorflow.keras.models import Model
conv2d_output = Model(inputs=model.input, outputs=model.get_layer('conv2d').output)
conv2d_1_output = Model(inputs=model.input,outputs=model.get_layer('conv2d_1').output)
conv2d_2_output = Model(inputs=model.input,outputs=model.get_layer('conv2d_2').output)
conv2d_3_output = Model(inputs=model.input,outputs=model.get_layer('conv2d_3').output)

conv2d_features = conv2d_output.predict(test_image)
conv2d_1_features = conv2d_1_output.predict(test_image)
conv2d_2_features = conv2d_2_output.predict(test_image)
conv2d_3_features = conv2d_3_output.predict(test_image)
print('First conv layer feature output shape : ',conv2d_features.shape)
print('Second conv layer feature output shape : ',conv2d_1_features.shape)
print('Third conv layer feature output shape : ',conv2d_2_features.shape)
print('Fourth conv layer feature output shape : ',conv2d_3_features.shape)

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

fig=plt.figure(figsize=(15,9))
columns = 8
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_1_features[0, :, :, i], cmap='gray')
plt.show()

fig=plt.figure(figsize=(15,9))
columns = 8
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_2_features[0, :, :, i], cmap='gray')
plt.show()

fig=plt.figure(figsize=(15,8))
columns = 8
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_3_features[0, :, :, i], cmap='gray')
plt.show()

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

from tensorflow.keras import backend as K

K.clear_session()
del model
