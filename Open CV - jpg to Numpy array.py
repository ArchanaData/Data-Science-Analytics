#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[3]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, cache_dir='.', untar = True)


# In[4]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)


# In[5]:


data_dir


# In[6]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[7]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[8]:


list(data_dir.glob('*/*.jpg'))


# In[9]:


list(data_dir.glob('*/*.jpg'))[:5]


# In[10]:


image_count = len(list(data_dir.glob('*/*.jpg')))
image_count


# In[11]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[12]:


roses = list(data_dir.glob('roses/*'))
roses[:5]
PIL.Image.open(str(roses[7]))


# In[13]:


roses = list(data_dir.glob('roses/*'))
roses[:5]


# In[14]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# In[15]:


flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*'))    
}


# In[16]:


flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


# In[17]:


flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4   
}


# In[18]:


flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# In[19]:


flowers_images_dict['roses'][0]


# In[20]:


str(flowers_images_dict['roses'][0])


# In[21]:


img = cv2.imread(str(flowers_images_dict['roses'][0]))
print(img)
print('\n',img.shape)


# In[22]:


img = cv2.imread(str(flowers_images_dict['roses'][0]))


# In[23]:


cv2.resize(img,(180,180)).shape


# In[24]:


X, y = [], []

for flower_name, images in flowers_images_dict.items():
    print(flower_name)
    print(len(images))


# In[25]:


for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


# In[26]:


X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


# In[27]:


y[:5]


# In[28]:


X[0]


# In[29]:


X = np.array(X)
y = np.array(y)


# In[30]:


0


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[32]:


X_train_scaled = X_train/255
X_test_scaled = X_test/255


# In[33]:


model = Sequential()

