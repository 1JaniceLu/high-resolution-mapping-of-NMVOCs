#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor


# In[ ]:


model=pickle.load(open("/out-of-sample/out-of-sample lgb.dat","rb"))


# # site

# In[ ]:


srcdir="/input/site all" 
for root,dirs,files in os.walk(srcdir):
    data=pd.DataFrame()
    for file in files:
        datai=pd.read_table(srcdir+os.sep+str(file),header=0,sep=",")
        data=pd.concat([data,datai],axis=1)


# In[ ]:


data["grid"].unique()


# In[ ]:


data.columns=['grid', 'doy', 'hour', 'distance', 'sate', 'ws', 'wd', 'tp', 'temp', 'solar',
       'pblh', 'ndvi', 'pop', 'dem', 'road', 'emission', 'landuse', 'o3','pm25']


# In[ ]:


data1=data[~data['wd'].isin([-9999])]
data1.describe().T


# In[ ]:


data1=data1[~data1['sate'].isin([-9999])]
data1.describe().T


# In[ ]:


#保存所有不含9999的行的索引
index_no9999=pd.DataFrame(data1.index)
index_no9999.columns=["number"]


# In[ ]:


data1=data1.reset_index(drop=True)
data1.drop('grid',axis=1,inplace=True)


# In[ ]:


ypreds=model.predict(data1)
ypreds=pd.DataFrame(ypreds)
ypreds.columns=["predicted"]
ypreds_all=pd.concat([index_no9999,ypreds],axis=1)
ypreds_all


# In[ ]:


pd1=pd.DataFrame()
pd1["number"]=np.arange(692625)  #692625 edited by true values


# In[ ]:


pd2=pd.merge(pd1,ypreds_all,how='outer')
pd2.fillna(-9999,inplace=True)
pd2["grid"]=data["grid"]
pd2.to_csv("/result/site pred voc.txt",header=True,index=False)


# # nosite

# reset the kernel

# In[ ]:


srcdir="/input/ID/nosite all" 
for root,dirs,files in os.walk(srcdir):
    data=pd.DataFrame()
    for file in files:
        datai=pd.read_table(srcdir+os.sep+str(file),header=0,sep=",")
        data=pd.concat([data,datai],axis=1)
data["number"]=data.index


# In[ ]:


data2=data1[~data1['sate'].isin([-9999])]   #sate: decided by true features
data2.min()


# In[ ]:


index1=pd.DataFrame(data2.loc[:,"number"])
data2.drop("number",axis=1,inplace=True)


# In[ ]:


ypreds=model.predict(data2)
ypreds=pd.DataFrame(ypreds)
ypreds.columns=["predicted"]
ypreds=pd.concat([index1,ypreds],axis=1)
ypreds=ypreds.reset_index(drop=True)
ypreds


# In[ ]:


#具体数字要根据行数更改
index=pd.DataFrame(np.arange(0,294919725,1))
index.columns=['number']
index


# In[ ]:


pd1=pd.merge(index,ypreds,how='outer')
pd1.fillna(-9999,inplace=True)
name="/result/nosite pred voc.txt"
pd1.to_csv(name,header=True,index=False)


# # merge

# In[ ]:


name="/result/site pred voc.txt"
zhandian=pd.read_table(name,header=0,sep=",")
name="/input/ID/site ID.txt"
ID_zhandian=pd.read_table(name,header=0,sep=",")
ID_zhandian.drop("grid",axis=1,inplace=True)
zhandian=pd.concat([ID_zhandian,zhandian],axis=1)
zhandian


# In[ ]:


#读取无站点预测数据集
name="/result/nosite pred voc.txt"
wuzhandian=pd.read_table(name,header=0,sep=",")
wuzhandian.drop("number",axis=1,inplace=True)
name="input/nosite ID.txt"
ID_wuzhandian=pd.read_table(name,header=0,sep=",")
wuzhandian=pd.concat([ID_wuzhandian,wuzhandian],axis=1)
wuzhandian


# In[ ]:


voc=pd.concat([zhandian,wuzhandian],axis=0)
voc=voc.set_index(voc["number"])
voc=voc.sort_index()
voc.drop("number",axis=1,inplace=True)


# In[ ]:


name="/result/pred voc(one col) nan-9999.txt"
voc.to_csv(name,header=True,index=False)


# # divide hourly data

# reset the kernel

# In[ ]:


name="/result/pred voc(one col) nan-9999.txt"
voc=pd.read_table(name,header=0,sep=",")
voc


# In[ ]:


name="input/date/only date(all grid).txt"
pd1=pd.read_table(name,header=0,sep=",")
voc=pd.concat([pd1,voc],axis=1)
groups=voc.groupby(voc.date)


# In[ ]:


name="/input/date/only date.xlsx"
onlydate=pd.read_excel(name,header=0,dtype=object)
onlydate["date"]=onlydate["date"].apply(str)
onlydate=onlydate["date"]


# In[ ]:


savefilepath="/result/onehour_excel/onecol"
m=1
for i in onlydate:
    zan=pd.DataFrame()
    zan=groups.get_group(i)
    zan.drop(["date"],axis=1,inplace=True)
    name=savefilepath+os.sep+str(m)+".xlsx"
    zan.to_excel(name,index=False,header=True)
    m+=1


# In[ ]:


index=pd.DataFrame(np.arange(1,98,1))
index.columns=['number']
index


# In[ ]:


pd1=pd.DataFrame()
for i in range(97):
    a=index.loc[i]
    pd1=pd1.append([a]*110)
pd1=pd1.reset_index(drop=True)


# In[ ]:


srcdir="/result/onehour_excel/onecol"
savefilepath="/result/onehour_excel/97row110col"


# In[ ]:


i=0
for root,dirs,files in os.walk(srcdir):
    for i in files:
        df1=pd.DataFrame()
        datai=pd.read_excel(srcdir+os.sep+i,header=0,dtype='str')
        datai["predicted"]=datai["predicted"].apply(str)
        datai
        df1=pd.concat([pd1,datai],axis=1)
        def func(df1):
            return ",".join(df1.values)
        y=df1.groupby(by="number").agg(func).reset_index()
        y2=pd.DataFrame()
        for p in range(110):
            zan=pd.DataFrame(y['predicted'].map(lambda x:x.split(',')[p]))
            y2=pd.concat([y2,zan],axis=1)
            p=p+1
        name=savefilepath+os.sep+str(i)
        y2.to_excel(name,index=False,header=False)


# In[ ]:




