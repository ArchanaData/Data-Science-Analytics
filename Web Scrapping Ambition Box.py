#!/usr/bin/env python
# coding: utf-8

# In[201]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


# In[202]:


requests.get('https://www.ambitionbox.com/list-of-companies?page=1').text


# In[203]:


headers1 = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
webpage1 = requests.get('https://www.ambitionbox.com/list-of-companies?page=1', headers=headers1).text


# In[204]:


soup1 = BeautifulSoup(webpage1, 'lxml')


# In[205]:


print(soup1.prettify())


# In[206]:


soup1.find_all('h1')


# In[207]:


soup1.find_all('h1')[0]


# In[208]:


soup1.find_all('h1')[0].text


# In[209]:


#soup.find_all('h2')

for ele in soup1.find_all('h2'):
    print(ele.text.strip())


# In[210]:


len(soup1.find_all('h2'))


# In[211]:


for i in soup1.find_all('p'):
    print(i.text.strip())


# In[212]:


for i in soup.find_all('p', class_='rating'):
    print(i.text.strip())


# In[213]:


len(soup.find_all('a', class_='review-count'))


# In[214]:


for i in soup.find_all('a', class_='review-count'):
    print(i.text.strip())


# In[215]:


len(soup.find_all('p', class_='infoEntity'))


# In[216]:


company = soup1.find_all('div', class_='company-content-wrapper')


# In[217]:


for i in company:
    print(i.text.strip())


# In[218]:


name = []
rating = []
review = []
sector = []
location = []
age = []
no_of_employees = []
about = []

for i in company:
    name.append(i.find('h2').text.strip())
    rating.append(i.find('p', class_='rating').text.strip())
    review.append(i.find('a', class_='review-count').text.strip())
    sector.append(i.find_all('p', class_='infoEntity')[0].text.strip())
    location.append(i.find_all('p', class_='infoEntity')[1].text.strip())
    age.append(i.find_all('p', class_='infoEntity')[2].text.strip())
    try:
        no_of_employees.append(i.find_all('p', class_='infoEntity')[3].text.strip())
    except:
        no_of_employees.append(np.nan)
    about.append(i.find('p', class_='description').text.strip())

age


# In[219]:


company


# In[220]:


d = {'name':name, 'rating':rating, 'review':review, 'sector':sector, 'location':location, 'age':age, 'no_of_employees':no_of_employees,'about':about}


# In[221]:


df = pd.DataFrame(d)
df


# In[ ]:



final = pd.DataFrame()
for j in range(1,17000):
    url = 'https://www.ambitionbox.com/list-of-companies?page={}'.format(j)
    headers1 = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
    webpage = requests.get(url , headers=headers).text
    
    soup = BeautifulSoup(webpage, 'lxml')
    
    company = soup.find_all('div', class_='company-content-wrapper')
    
    name = []
    rating = []
    review = []
    sector = []
    location = []
    age = []
    no_of_employees = []
    about = []

    for i in company:
        try:
            name.append(i.find('h2').text.strip())
        except:
            name.append(np.nan)
        try:
            rating.append(i.find('p', class_='rating').text.strip())
        except:
            rating.append(np.nan)
        try:
            review.append(i.find('a', class_='review-count').text.strip())
        except:
            review.append(np.nan)
        try:
            sector.append(i.find_all('p', class_='infoEntity')[0].text.strip())
        except:
            sector.append(np.nan)
        try:
            location.append(i.find_all('p', class_='infoEntity')[1].text.strip())
        except:
            location.append(np.nan)
        try:
            age.append(i.find_all('p', class_='infoEntity')[2].text.strip())
        except:
            age.append(np.nan)
        try:
            no_of_employees.append(i.find_all('p', class_='infoEntity')[3].text.strip())
        except:
            no_of_employees.append(np.nan)
        try:
            about.append(i.find('p', class_='description').text.strip())
        except:
            about.append(np.nan)

    d = {'name':name, 'rating':rating, 'review':review, 'sector':sector, 'location':location, 'age':age, 'no_of_employees':no_of_employees,'about':about}
    df = pd.DataFrame(d)
    final = final.append(df)


# In[223]:


final


# In[ ]:




