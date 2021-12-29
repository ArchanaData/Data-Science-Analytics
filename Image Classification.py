#!/usr/bin/env python
# coding: utf-8

# In[69]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[55]:


(X_train, y_train), (X_test, y_test) =  datasets.cifar10.load_data()
X_train.shape


# In[56]:


X_test.shape


# In[57]:


y_train[:5] #2D array


# In[58]:


y_train = y_train.reshape(-1,)
y_train[:5] #1D array


# In[59]:


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[60]:


def plot_sample(X,y,index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[61]:


plot_sample(X_train, y_train, 1111)


# In[62]:


X_train = X_train/255
X_test = X_test/255


# In[63]:


# Artificial Neural Network
ann = models.Sequential([
    layers.Flatten(input_shape = (32,32,3)),
    layers.Dense(3000, activation = 'relu'),
    layers.Dense(1000, activation = 'relu'),
    layers.Dense(10, activation = 'sigmoid')    
])

ann.compile(optimizer = 'SGD', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, epochs = 5)


# In[64]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]


# In[65]:


print("Classification report: \n ", classification_report(y_test, y_pred_classes))


# In[66]:


# Convulation Neural Network
cnn = models.Sequential([
    #cnn
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    
    #dense
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10, activation = 'softmax') #softmax is used to normalize probability    
])


# In[67]:


cnn.compile(optimizer = 'adam',
           loss = 'sparse_categorical_crossentropy',
           metrics = ['accuracy'])


# In[68]:


cnn.fit(X_train, y_train, epochs = 10)


# In[70]:


cnn.evaluate(X_test, y_test)


# In[71]:


X_test[:5]


# In[ ]:




