#!/usr/bin/env python
# coding: utf-8

# # Data preprocessing

# In[ ]:


import os
import pandas as pd
import numpy as np


# ## deal with every variables 

# In[ ]:


name="/input/NDVI/all NDVI.txt"   #have two col "grid” and "value"
data=pd.read_table(name,header=0,sep=',')
data


# In[ ]:


nums=[1047,1155,2020,2376,2594,
         2597,2788,2820,3043,4005,
         5704,5814,5815,6000,6035,
         7980,8209,8306,8309,8318,
         8424,8426,8641,8645,8965,
         8970,8972,9068,9077,9290,
         9502,9613,9615,9723,9725,
         9728,9833,9834,9835,9838,
         9943,10057,10058]
zhandian=pd.DataFrame()
for num in nums:
    zan=data.loc[data["grid"]==num,:]
    zhandian=pd.concat([zhandian,zan],axis=0)
zhandian=zhandian.drop("grid",axis=1)
name="/input/NDVI/site NDVI.txt"
zhandian.to_csv(name,header=True,index=False)


# In[ ]:


for num in nums:
    data.drop(data.index[(data["grid"]==num)],inplace=True)
name="/input/NDVI/nosite NDVI.txt"
data.to_csv(name,header=True,index=False)


# # deal with train data

# In[ ]:


srcdir="/input/site all"  #without voc 
for root,dirs,files in os.walk(srcdir):
    data=pd.DataFrame()
    for file in files:
        datai=pd.read_table(srcdir+os.sep+str(file),header=0,sep=",")
        data=pd.concat([data,datai],axis=1)
data


# In[ ]:


voc=pd.read_excel("input/voc/voc edit.xlsx",header=0,index_col=None)
voc=voc.reset_index(drop=True)
pd1=pd.concat([pd1,voc.iloc[:,-1]],axis=1)
pd2=pd1.dropna()


# In[ ]:


pd3=pd2[~pd2['10v'].isin([-9999])]   #"10v" decided by true features
pd3.min()


# In[ ]:


pd3.columns


# In[ ]:


#获取风向和风量
deg=180.0/np.pi
pd3["ws"]=3.16228*np.hypot(pd3["10u"],pd3["10v"])
pd3["wd"]= 180.0 + np.arctan2(pd3["10u"], pd3["10v"])*deg
pd3.drop(["10u","10v"],axis=1,inplace=True)


# In[ ]:


pd3.columns=['grid', 'doy', 'hour', 'distance', 'sate', 'ws', 'wd', 'tp', 'temp', 'solar',
       'pblh', 'ndvi', 'pop', 'dem', 'road', 'emission', 'landuse', 'voc']


# In[ ]:


pd3.to_excel("/tune/out-of-station/out-of-station data.xlsx",header=True,index=False)


# In[ ]:


pd3.drop("grid",axis=1,inplace=True)
pd3.to_excel("/tune/out-of-sample/out-of-sample data.xlsx",header=True,index=False)


# In[ ]:




