#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf


# In[22]:


get_ipython().system('pip install matplotlib')


# In[24]:


import matplotlib.pyplot as plt


# In[4]:


mnist=tf.keras.datasets.mnist


# In[5]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[6]:


x_train


# In[25]:


plt.imshow(x_train[5],cmap=plt.cm.binary)
plt.show()


# In[ ]:





# In[7]:


x_train=tf.keras.utils.normalize(x_train,axis=1)
x_train


# In[9]:


x_train=tf.keras.utils.normalize(x_train,axis=1)
x_train


# In[8]:


x_test=tf.keras.utils.normalize(x_test,axis=1)


# In[9]:


x_test


# In[10]:


x_train[0]


# In[11]:


x_test[0]


# In[12]:


models=tf.keras.models.Sequential()
787


# In[13]:


models.add(tf.keras.layers.Flatten()) # flats the input takes 28*28 and produces 1*784


# In[14]:


models.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
models.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


# In[15]:


models.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
models.fit(x_train,y_train,epochs=10)


# In[17]:


value_loss,value_accuracy=models.evaluate(x_test,y_test)


# In[18]:


value_loss


# In[19]:


value_accuracy


# In[21]:


models.save(r'E:\digital classification\1stdigitmodel')


# In[26]:


get_ipython().system('pip install numpy as np')


# In[27]:


new_model=tf.keras.models.load_model(r'E:\digital classification\1stdigitmodel')


# In[28]:


predictions=new_model.predict(x_test)


# In[29]:


predictions[0]


# In[30]:


plt.imshow(x_test[40],cmap=plt.cm.binary)
plt.show()


# In[31]:


import numpy as np


# In[32]:


np.argmax(predictions[40])


# In[33]:


plt.imshow(x_test[87],cmap=plt.cm.binary)
plt.show()


# In[34]:


np.argmax(predictions[87])


# In[35]:


plt.imshow(x_test[78],cmap=plt.cm.binary)
plt.show()


# In[38]:


np.argmax(predictions[87])


# In[39]:


plt.imshow(x_test[88],cmap=plt.cm.binary)
plt.show()


# In[41]:


np.argmax(predictions[88])


# In[ ]:




