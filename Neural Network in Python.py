#!/usr/bin/env python
# coding: utf-8

# In[46]:


import tensorflow


# In[47]:


url = 'https://raw.githubusercontent.com/codebasics/deep-learning-keras-tf-tutorial/master/6_gradient_descent/insurance_data.csv'


# In[48]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[49]:


df = pd.read_csv(url)


# In[50]:


df.rename(columns = {'affordibility':'affordability'}, inplace = True)


# In[51]:


df


# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(df[['age', 'affordability']], df.bought_insurance, test_size = 0.2, random_state = 25)


# In[54]:


X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age']/100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100


# In[55]:


model = keras.Sequential([
    keras.layers.Dense(1, input_shape = (2,), activation = 'sigmoid', kernel_initializer = 'ones', bias_initializer = 'zeros')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[56]:


model.fit(X_train_scaled, y_train, epochs = 5000)


# In[57]:


model.evaluate(X_test_scaled, y_test)


# In[58]:


model.predict(X_test_scaled)


# In[59]:


y_test


# In[60]:


coef, intercept = model.get_weights()


# In[61]:


coef, intercept


# In[62]:


def sigmoid(x):
    import math
    return 1/(1 + math.exp(-x))
sigmoid(5)


# In[63]:


X_test


# In[64]:


def prediction_function(age, affordability):
    weighted_sum = coef[0]*age + coef[1]*affordability + intercept
    return sigmoid(weighted_sum)

prediction_function(.28,1)


# # Implement NN in plain python

# In[65]:


def sigmoid_numpy(X):
    return 1/(1+np.exp(-X))
sigmoid_numpy(np.array([12,0,2]))


# In[97]:


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*np.log(1-y_predicted_new))


# In[105]:


class myNN:
    def __init__ (self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
    
    def fit(self, X, y, epochs, loss_threshold):
        self.w1, self.w2, self.bias, self.loss = self.gradient_descent(X['age'], X['affordability'], y_train, epochs, loss_threshold)
        print(f'The final weights & bias: w1:{self.w1}, w2:{self.w2}, bias:{self.bias}, loss:{self.loss}')
    
    def predict(self, X_test):
        weighted_sum = self.w1*X_test['age'] + self.w2*X_test['affordability'] + self.bias
        return sigmoid_numpy(weighted_sum)
    
    
    def gradient_descent(self,age,affordability, y_true, epochs, loss_threshold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1*age + w2*affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)

            w1d = (1/n)*np.dot(np.transpose(age), (y_predicted-y_true))
            w2d = (1/n)*np.dot(np.transpose(affordability), (y_predicted-y_true))

            bias_d = np.mean(y_predicted-y_true)

            w1 = w1 - rate*w1d
            w2 = w2 - rate*w2d
            bias = bias - rate*bias_d

            if i%50==0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

            if loss<=loss_threshold:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1,w2,bias,loss


# In[106]:


custom_model = myNN()
custom_model.fit(X_train_scaled, y_train, epochs = 8000, loss_threshold = 0.4631 )


# In[107]:


coef, intercept


# In[108]:


X_test_scaled


# In[109]:


custom_model.predict(X_test_scaled)


# In[110]:


y_test


# In[111]:


model.predict(X_test_scaled)


# In[ ]:




