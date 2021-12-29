#!/usr/bin/env python
# coding: utf-8
url: https://api.themoviedb.org/3/movie/popular?api_key=<<api_key>>&language=en-US&page=1API key: 6e3cfa534dd1d7ae5bb2526f44327578
# Columns: Popularity,vote count,original lang,title,vote avg,overview,release date (create df of shape 10000*7)

# In[30]:


import requests
import pandas as pd
import numpy as np
response = requests.get('https://api.themoviedb.org/3/movie/popular?api_key=6e3cfa534dd1d7ae5bb2526f44327578&language=en-US&page=10').json()


# In[14]:


response


# In[15]:


response['total_results']


# In[16]:


response['total_pages']


# In[18]:


response['results'] # cuz this is an array


# In[27]:


d1 = {'name':['Baba', 'Beti'], 'age':['infinity',32]}
df1 = pd.DataFrame(d)
df1


# In[28]:


popularity = []
vote_count = []
original_language = []
title = []
vote_average = []
overview = []
release_date = []

for i in response['results']:
    popularity.append(i['popularity'])
    vote_count.append(i['vote_count'])
    original_language.append(i['original_language'])
    title.append(i['title'])
    vote_average.append(i['vote_average'])
    overview.append(i['overview'])
    release_date.append(i['release_date'])
    
d = {'title':title, 'overview':overview, 'original_language':original_language, 'release_date':release_date, 'popularity':popularity, 'vote_count':vote_count, 'vote_average':vote_average}

df = pd.DataFrame(d)
df


# In[31]:


final = pd.DataFrame()

for j in range(1,501):
    response = requests.get('https://api.themoviedb.org/3/movie/popular?api_key=6e3cfa534dd1d7ae5bb2526f44327578&language=en-US&page={}'.format(j)).json()
    popularity = []
    vote_count = []
    original_language = []
    title = []
    vote_average = []
    overview = []
    release_date = []

    for i in response['results']:
        try:
            popularity.append(i['popularity'])
        except:
            popularity.append(np.nan)
        try:
            vote_count.append(i['vote_count'])
        except:
            vote_count.append(np.nan)
        try:
            original_language.append(i['original_language'])
        except:
            original_language.append(np.nan)
        try:
            title.append(i['title'])
        except:
            title.append(np.nan)
        try:
            vote_average.append(i['vote_average'])
        except:
            vote_average.append(np.nan)
        try:
            overview.append(i['overview'])
        except:
            overview.append(np.nan)
        try:
            release_date.append(i['release_date'])
        except:
            release_date.append(np.nan)

    d = {'title':title, 'overview':overview, 'original_language':original_language, 'release_date':release_date, 'popularity':popularity, 'vote_count':vote_count, 'vote_average':vote_average}

    df = pd.DataFrame(d)
    
    final = final.append(df, ignore_index = True)


# In[33]:


final.shape


# In[34]:


final.head()


# In[35]:


final.to_csv('movies.csv')


# In[ ]:




