
# coding: utf-8

# # Homes Regression

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[3]:


data = pd.read_csv('cal_housing_clean.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X = data.drop('medianHouseValue',axis=1)
y = data['medianHouseValue']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[10]:


from sklearn.preprocessing import MinMaxScaler


# In[11]:


mms = MinMaxScaler()


# In[12]:


mms.fit(X_train)


# In[13]:


X_train = pd.DataFrame(mms.transform(X_train),columns=X_train.columns,index=X_train.index)


# In[14]:


X_test = pd.DataFrame(mms.transform(X_test),columns=X_test.columns,index=X_test.index)


# In[15]:


age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
household = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')


# In[16]:


feature_cols = [age,rooms,bedrooms,pop,household,income]


# In[17]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train, batch_size=10, num_epochs=1000, shuffle=True)


# In[18]:


model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feature_cols)


# In[19]:


model.train(input_fn=input_func,steps=25000)


# In[20]:


pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)


# In[21]:


pred_gen = model.predict(input_fn=pred_input_func)
predictions = list(pred_gen)


# In[22]:


list_pred = []
for item in predictions:
    list_pred.append(item['predictions'])


# In[23]:


final_pred = []
for item in list_pred:
    final_pred.append(item[0])
final_pred


# In[24]:


from sklearn.metrics import mean_squared_error


# In[25]:


np.sqrt(mean_squared_error(y_test,final_pred))

