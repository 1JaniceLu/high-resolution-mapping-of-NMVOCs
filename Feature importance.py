#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import shap
import matplotlib.pylab as pl
shap.initjs()
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import math 
from math import sqrt
import lightgbm as lgb


# In[ ]:


data=pd.read_excel("/tune/out-of-sample/out-of-sample data.xlsx",header=0,index_col=None)
data


# In[ ]:


X = data.columns[:-1]
Y = data.columns[-1] 


# In[ ]:


model=pickle.load(open("/tune/out-of-sample/out-of-sample stlgb.dat","rb"))


# In[ ]:


explainer=shap.TreeExplainer(model)


# In[ ]:


shap_values1=explainer.shap_values(X,check_additivity=False) 


# In[ ]:


feature_importance=pd.DataFrame()
feature_importance['feature']=x_train_data.columns
feature_importance['importance']=np.abs(shap_values1).mean(0)
feature_importance.sort_values('importance',ascending=False)
feature_importance.to_excel("/feature importance/feature importance.xlsx",index=False,header=True)


# In[ ]:


#summary plot
fig=shap.summary_plot(shap_values1,X,max_display=25,show=False)
pl.xlim(-50,200)
pl.xticks(fontsize=15)
pl.yticks(fontsize=15)
pl.xlabel('SHAP value',fontsize=20)
pl.tight_layout() 
pl.savefig('/shap/summaryplot.png',dpi=300)


# In[ ]:


shap_interaction_values1 = explainer.shap_interaction_values(X)   #very slow,may cut data


# In[ ]:


#main depeandence plot
fig=shap.dependence_plot(("emsi","emsi")
              ,shap_interaction_values1,X
              ,show=False)
pl.xlim(278,370) 
pl.ylim(-60,60)
ax=pl.gca()
x_major_locator=MultipleLocator(20)
y_major_locator=MultipleLocator(20)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
pl.xticks(fontsize=15)
pl.yticks(fontsize=15)
pl.xlabel("Emission",fontsize=20)
pl.ylabel("SHAP main effect value for\nEmission")
pl.tight_layout()
pl.savefig("/shap/main effect emission.png",dpi=300)


# In[ ]:


#interaction dependence plot
shap.dependence_plot(("ndvi","pop"),shap_interaction_values1,X
              ,show=False)
pl.xlim(-0.5,8500)
pl.ylim(-20,30)
ax=pl.gca()
x_major_locator=MultipleLocator(2000)
y_major_locator=MultipleLocator(10)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
pl.xticks(fontsize=15)
pl.yticks(fontsize=15)
pl.xlabel("NDVI",fontsize=20)
pl.ylabel("SHAP value",fontsize=20)
pl.tight_layout()
pl.savefig("/shap/interact ndvi(with pop).png",dpi=300)

