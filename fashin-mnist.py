#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('../../datasets/fashion-mnist/fashion-mnist_train.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


y = df['label']
X = df.drop('label',axis=1)


# In[6]:


y.value_counts()


# Each training and test example is assigned to one of the following labels:
# 
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

# In[17]:


plt.figure(figsize=(1,1))
plt.imshow(X.iloc[0].values.reshape(28,28))


# In[18]:


X = X/255


# In[19]:


from tensorflow.keras.utils import to_categorical


# In[20]:


y.shape


# In[21]:


y = to_categorical(y,num_classes=10)
y.shape


# In[23]:


X.shape


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[27]:


model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(784,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


# In[28]:


model.summary()


# In[29]:


hist = model.fit(X,y,epochs=30,batch_size=64)


# In[31]:


plt.plot(hist.history['loss'])


# In[42]:


img = X.iloc[4].values.reshape(1,784)
model.predict_on_batch(img).argmax()


# In[39]:


df['label']


# In[ ]:




