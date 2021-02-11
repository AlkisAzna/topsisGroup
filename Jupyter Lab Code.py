#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import math

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# In[2]:


# Data candidates
df_candidates = pd.read_excel("Candidates_Results.xlsx").iloc[:-1]
df_persons = df_candidates.astype('int').set_index('No')
df_persons


# In[3]:


#Interview
df_interview = pd.read_excel("Interview_Results.xlsx", header=1).set_index('No')
df_interview


# In[4]:


#Weights
df_weights_per_DM = pd.read_excel("Weights_Matrix.xlsx").iloc[:7].drop(['No'], axis = 1).set_index('Attributes')
df_weights_per_DM


# In[5]:


total_DMs = df_weights_per_DM.columns.size


# In[6]:


decision_matrix = {}
for x in range(total_DMs):
    decision_matrix["DM"+str(x+1)] = df_persons.copy()
decision_matrix


# In[7]:


for x in range(total_DMs):
    if x==0:
        decision_matrix["DM"+str(x+1)][['Panel Interview', '1-on-1 Interview']] = df_interview[['Panel Interview', '1-on-1 Interview']]
    else:
        decision_matrix["DM"+str(x+1)][['Panel Interview', '1-on-1 Interview']] = df_interview[['Panel Interview.'+str(x), '1-on-1 Interview.'+str(x)]]
decision_matrix


# In[8]:


normalized_matrix = decision_matrix.copy()


# In[9]:


# df['column name'] = df['column name'].replace(['old value'],'new value')
for x in range(total_DMs):
    for idx, col in enumerate(decision_matrix["DM"+str(x+1)].columns):
        sum_of_squares = math.sqrt(float(decision_matrix["DM"+str(x+1)].pow(2).sum().get(idx)))
        for index, row in enumerate(decision_matrix["DM"+str(x+1)].index):
            curr_value = decision_matrix["DM"+str(x+1)].loc[row,col]
            normalized_value= curr_value/sum_of_squares
            normalized_matrix["DM"+str(x+1)].iloc[index, normalized_matrix["DM"+str(x+1)].columns.get_loc(col)] = normalized_value
normalized_matrix


# In[10]:


positive_ideal_sol = {}
negative_ideal_sol = {}
for x in range(total_DMs):
    positive_ideal_sol["DM"+str(x+1)] = np.empty(normalized_matrix["DM"+str(x+1)].columns.size, dtype=float)
    negative_ideal_sol["DM"+str(x+1)] = np.empty(normalized_matrix["DM"+str(x+1)].columns.size, dtype=float) 
    for idx, col in enumerate(normalized_matrix["DM"+str(x+1)].columns):
        max_value = max(normalized_matrix["DM"+str(x+1)][col])
        min_value = min(normalized_matrix["DM"+str(x+1)][col])
        positive_ideal_sol["DM"+str(x+1)][idx] = max_value
        negative_ideal_sol["DM"+str(x+1)][idx] = min_value
print(positive_ideal_sol)
print(negative_ideal_sol)


# In[99]:


#Euclidean distance p = 2 and Manhattan p = 1
euclidean_measure_pis = {}
manhattan_measure_pis = {}
euclidean_measure_nis = {}
manhattan_measure_nis = {}
for x in range(total_DMs):
    euclidean_measure_pis["DM"+str(x+1)] = np.empty(normalized_matrix["DM"+str(x+1)].index.size, dtype=float)
    manhattan_measure_pis["DM"+str(x+1)] = np.empty(normalized_matrix["DM"+str(x+1)].index.size, dtype=float)
    euclidean_measure_nis["DM"+str(x+1)] = np.empty(normalized_matrix["DM"+str(x+1)].index.size, dtype=float)
    manhattan_measure_nis["DM"+str(x+1)] = np.empty(normalized_matrix["DM"+str(x+1)].index.size, dtype=float)
    for index, row in enumerate(normalized_matrix["DM"+str(x+1)].index):
        temp_euclidean_pis = 0.0
        temp_euclidean_nis = 0.0
        temp_manhattan_pis = 0.0
        temp_manhattan_nis = 0.0
        temp_value = 0.0
        for idx, col in enumerate(normalized_matrix["DM"+str(x+1)].columns):
            curr_weight = df_weights_per_DM.loc[col,"DM"+str(x+1)]
            curr_pis = positive_ideal_sol["DM"+str(x+1)][idx]
            curr_nis = negative_ideal_sol["DM"+str(x+1)][idx]
            curr_value = normalized_matrix["DM"+str(x+1)].loc[row,col]
            # Euclidean
            temp_value = curr_weight*((curr_value-curr_pis) ** 2)
            temp_euclidean_pis = temp_euclidean_pis + temp_value
            temp_value = curr_weight*((curr_value-curr_nis) ** 2)
            temp_euclidean_nis = temp_euclidean_nis + temp_value
            # Manhattan
            temp_value = curr_weight*(curr_pis - curr_value)
            temp_manhattan_pis = temp_manhattan_pis + temp_value
            temp_value = curr_weight*(curr_value-curr_nis)
            temp_manhattan_nis = temp_manhattan_nis + temp_value
        euclidean_measure_pis["DM"+str(x+1)][index] = math.sqrt(temp_euclidean_pis)
        manhattan_measure_pis["DM"+str(x+1)][index] = temp_manhattan_pis
        euclidean_measure_nis["DM"+str(x+1)][index] = math.sqrt(temp_euclidean_nis)
        manhattan_measure_nis["DM"+str(x+1)][index] = temp_manhattan_nis
print(manhattan_measure_nis)


# In[100]:


eucl_pis_arithmetic = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
eucl_pis_geometric = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
manhattan_pis_arithmetic = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
manhattan_pis_geometric = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
eucl_nis_arithmetic = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
eucl_nis_geometric = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
manhattan_nis_arithmetic = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
manhattan_nis_geometric = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
for index, row in enumerate(normalized_matrix["DM1"].index):
    temp_pow = 1 / total_DMs
    #Arithmetic mean
    #PIS
    temp_value = 0
    for x in range(total_DMs):
        temp_value = temp_value + euclidean_measure_pis["DM" + str(x+1)][index]
    eucl_pis_arithmetic[index] = temp_value / total_DMs
    temp_value = 0
    
    for x in range(total_DMs):
        temp_value = temp_value + manhattan_measure_pis["DM" + str(x+1)][index]
    manhattan_pis_arithmetic[index] = temp_value / total_DMs
    #NIS
    temp_value = 0
    for x in range(total_DMs):
        temp_value = temp_value + euclidean_measure_nis["DM" + str(x+1)][index]
    eucl_nis_arithmetic[index] = temp_value / total_DMs
    temp_value = 0
    
    for x in range(total_DMs):
        temp_value = temp_value + manhattan_measure_nis["DM" + str(x+1)][index]
    manhattan_nis_arithmetic[index] = temp_value / total_DMs
    
    #Geometric mean
    #PIS
    temp_value = 1
    for x in range(total_DMs):
        temp_value = temp_value * euclidean_measure_pis["DM" + str(x+1)][index]
    eucl_pis_geometric[index] = temp_value ** temp_pow
    temp_value = 1
    for x in range(total_DMs):
        temp_value = temp_value * manhattan_measure_pis["DM" + str(x+1)][index]
    manhattan_pis_geometric[index] = temp_value ** temp_pow
    #NIS
    temp_value = 1
    for x in range(total_DMs):
        temp_value = temp_value * euclidean_measure_nis["DM" + str(x+1)][index]
    eucl_nis_geometric[index] = temp_value ** temp_pow
    temp_value = 1
    for x in range(total_DMs):
        temp_value = temp_value * manhattan_measure_nis["DM" + str(x+1)][index]
    manhattan_nis_geometric[index] = temp_value ** temp_pow


# In[103]:


rel_close_eucl_arithmetic = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
rel_close_eucl_geometric = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
rel_close_manh_arithmetic = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
rel_close_manh_geometric = np.empty(normalized_matrix["DM1"].index.size, dtype=float)
for index, row in enumerate(normalized_matrix["DM1"].index):
    rel_close_eucl_arithmetic[index] = eucl_nis_arithmetic[index] / (eucl_nis_arithmetic[index] + eucl_pis_arithmetic[index])
    rel_close_eucl_geometric[index] = eucl_nis_geometric[index] / (eucl_nis_geometric[index] + eucl_pis_geometric[index])
    rel_close_manh_arithmetic[index] = manhattan_nis_arithmetic[index] / (manhattan_nis_arithmetic[index] + manhattan_pis_arithmetic[index])
    rel_close_manh_geometric[index] = manhattan_nis_geometric[index] / (manhattan_nis_geometric[index] + manhattan_pis_geometric[index])
rel_close_manh_geometric


# In[104]:


# Ranks Euclidean-Arithmetic
array1 = rel_close_eucl_arithmetic.copy()
temp = (-array1).argsort()
ranks_eucl_arithmetic = np.arange(len(array1))[temp.argsort()] + 1
print(ranks_eucl_arithmetic)

# Ranks Euclidean-Geometric
array2 = rel_close_eucl_geometric.copy()
temp = (-array2).argsort()
ranks_eucl_geometric = np.arange(len(array2))[temp.argsort()] + 1
print(ranks_eucl_geometric)

# Ranks Manhattan-Arithmetic
array3 = rel_close_manh_arithmetic.copy()
temp = (-array3).argsort()
ranks_manh_arithmetic = np.arange(len(array3))[temp.argsort()] + 1
print(ranks_manh_arithmetic)

# Ranks Manhattan-Geometric
array4 = rel_close_manh_geometric.copy()
temp = (-array4).argsort()
ranks_manh_geometric = np.arange(len(array4))[temp.argsort()] + 1
print(ranks_manh_geometric)
