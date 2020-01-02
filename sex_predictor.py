#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importando as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


#Chamando o dataset
data = pd.read_csv('test_data_CANDIDATE.csv')
data2 = pd.read_csv('newsample.csv')


# In[3]:


# Preenche os dados missing de chol com a media
data.chol.fillna(data.chol.mean(), inplace = True)
data2.chol.fillna(data.chol.mean(), inplace = True)


# In[4]:


#Retirando os dados 
data.drop(columns=['cp','slope','index'], axis = 1, inplace = True)
data2.drop(columns=['cp','slope','index'], axis = 1, inplace = True)


# In[5]:


#Colocando os caracteres F e M em maiúscula
def upper(df, col):
    return data.assign(**{col : df[col].str.upper()})

data = upper(data, "sex")


# In[ ]:


#Transformando os dados trf de segundos para horas
trf2 = data.trf/3600
trf3 = data2.trf/3600
data.drop('trf', axis = 1, inplace = True)
data2.drop('trf', axis = 1, inplace = True)
data = pd.concat([data,trf2], axis = 1, join = 'inner')
data2 = pd.concat([data2,trf3], axis = 1, join = 'inner')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
#Obtendo as colunas categóricas
cat_col = [var for var in data.columns if data[var].dtype == 'object']
df = data.loc[:,cat_col]

#codificando a coluna sex sendo M(1) e F(0).
encoder = LabelEncoder()
data_int = df.apply(encoder.fit_transform)

data_f = data.drop('sex', axis = 1)

uniao = [data_int, data_f]
data_final = pd.concat(uniao, axis = 1, join = 'inner')


# In[ ]:


data_final.head()


# In[ ]:


#Arrumando o desbalanceamento
from imblearn.under_sampling import RandomUnderSampler
# Number of data points in the minority class
number_records_f = len(data_final[data_final.sex == 0])
f_indices = np.array(data_final[data_final.sex == 0].index)

# Picking the indices of the normal classes
normal_indices = data_final[data_final.sex == 1].index

# Out of the indices we picked, randomly select "x" number (number_records_f)
random_normal_indices = np.random.choice(normal_indices, number_records_f)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([f_indices,random_normal_indices])


# Under sample dataset
under_sample_data = data_final.loc[under_sample_indices,:]

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'sex']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'sex']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.sex == 1])/len(under_sample_data))
print("Percentage of f transactions: ", len(under_sample_data[under_sample_data.sex == 0])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# In[ ]:


#Separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size = 0.2)


# In[ ]:


from sklearn.svm import SVC
clf = SVC(gamma = 'auto')
clf.fit(X_train, y_train)
resultado_svc = clf.predict(data2)
resultado_svc


# In[ ]:


# Criando dataset de submissao
final = encoder.inverse_transform(resultado_svc)
submission = pd.DataFrame({ "sex": final.reshape((final.shape[0]))})

print(submission)


# In[ ]:


submission.to_csv('newsample_PREDICTIONS_{Juliana Facchini de Souza}.csv', index=False)

