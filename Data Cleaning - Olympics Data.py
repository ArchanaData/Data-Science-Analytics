#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("athlete_events.csv")
region_df = pd.read_csv("noc_regions.csv")


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df = df[df["Season"]=="Summer"]


# In[6]:


df.shape


# In[7]:


df.tail()


# In[8]:


df = df.merge(region_df, on = "NOC", how = "left")


# In[9]:


df.tail()


# In[10]:


df['region'].unique().shape


# In[11]:


df.isnull().sum()


# In[12]:


df.duplicated().sum()


# In[13]:


df.drop_duplicates(inplace = True)


# In[14]:


df.duplicated().sum()


# In[15]:


df['Medal'].value_counts() 


# In[16]:


# doing one hot encoding to remove NaN values
pd.get_dummies(df['Medal'])


# In[17]:


df.shape


# In[18]:


# The number of cols are same in dummy & df
df = pd.concat([df, pd.get_dummies(df['Medal'])], axis = 1)


# In[19]:


df.groupby('NOC').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending = False)


# In[20]:


df.groupby('NOC').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending = False).reset_index()


# In[21]:


# medal count is very high for all countries, because team match is considering gold medal for every player of the team
df[(df['NOC']=='IND') & (df['Medal'] == 'Gold')]


# In[22]:


medal_tally = df.drop_duplicates(subset = ['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])


# In[23]:


medal_tally


# In[24]:


medal_tally = medal_tally.groupby('NOC').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending = False).reset_index()


# In[25]:


medal_tally


# In[26]:


medal_tally[medal_tally['NOC']=='IND']


# In[ ]:




