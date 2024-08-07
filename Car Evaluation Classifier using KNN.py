#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn

from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
data= pd.read_csv(r'car.data')
print(data.head())
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
cls = le.fit_transform(list(data["class"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
X = list(zip(buying,door,maint,persons,lug_boot))
y = list(cls)
X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)
model = KNeighborsClassifier(n_neighbors= 11)
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)
print(acc)
prediction = model.predict(X_test)
names = ('unacc','acc','good','vgood')
for x in range(len(prediction)):
    print("prediction : ",names[prediction[x]] , "  data : ",X_test[x]," actual : ",names [y_test[x]])


# In[ ]:




