#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import folktables
from folktables import ACSDataSource, ACSEmployment, ACSIncome
from folktables.acs import adult_filter, public_coverage_filter, travel_time_filter


# In[39]:


data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=None, density=0.02, download=True)


# In[52]:


feature_type = {
    ## income
    'COW': 'dis',
    'SCHL': 'num',
    'MAR': 'dis',
    'OCCP': 'dis',
    'POBP': 'dis',
    'RELP': 'dis',
    'WKHP': 'num',
    
    ## employ
    'DIS': 'dis',
    'ESP': 'dis',
    'CIT': 'dis',
    'MIG': 'dis',
    'MIL': 'dis',
    'ANC': 'dis',
    'NATIVITY': 'dis',
    'DEAR': 'dis',
    'DEYE': 'dis',
    'DREM': 'dis',
    
    ## public coverage
    'PINCP': 'num',
    'ESR': 'dis',
    'ST': 'dis',
    'FER': 'dis', ## !!! gender specific, giving birth or not !!!
    
    ## mobility
    'GCL': 'dis',
    'JWMNP': 'num',
    'POVPIP': 'num',
    
    ## travel time
    "JWTR": 'dis',
    
    ## sensitive attribute
    'AGEP': 'num',
    'SEX': 'dis',
    'RAC1P': 'dis',
}
for f in feature_type:
    assert feature_type[f] in ['dis', 'num']


# In[53]:


def my_one_hot(a):
    values = np.unique(a)
    index_dict = {values[i]:i for i in range(len(values))}
    mapper = np.vectorize(lambda x : index_dict[x])
    
    #print(values, index_dict)
    
    a1 = np.zeros((a.size, len(index_dict)))
    a1[np.arange(a.size),mapper(a)] = 1
    
    #print(a1)
    return a1[:, 1:]

def z_trans(a):
    mean = np.mean(a)
    std = np.std(a)
    return (a-mean)*1./std


# In[59]:


def make_data(task, data, path):
    print(f"Save to path {path}")
    
    X, y, z = task.df_to_numpy(data)

    X_1 = np.zeros((X.shape[0], 0))
    z_1 = np.zeros((z.shape[0], 0))

    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")
    print(f"z.shape = {z.shape}")

    ## preprocess X
    for j in range(X.shape[1]):
        f = task.features[j]
        
        assert f in feature_type, f
        
        if(f in feature_mapper):
            mapper = np.vectorize(feature_mapper[f])
            X[:, j] = mapper(X[:, j])
        if(feature_type[f] == 'dis'):
            X_1 = np.hstack((X_1, my_one_hot(X[:, j])))
        else:
            X_1 = np.hstack((X_1, z_trans(X[:, j]).reshape(-1, 1)))
#         else(feature_type[f] == 'num'):
#             X_1 = np.hstack((X_1, z_trans(X[:, j]).reshape(-1, 1)))
#         else:
#             X_1 = np.hstack((X_1, X[:, j].reshape(-1, 1)))
            
    ## preprocess z        
    for j in range(z.shape[1]):
        f = task.group[j]
        if(f in feature_mapper):
            mapper = np.vectorize(feature_mapper[f])
            z[:, j] = mapper(z[:, j])
        if(feature_type[f] == 'dis'):
            z_1 = np.hstack((z_1, my_one_hot(z[:, j])))
        else:             
            z_1 = np.hstack((z_1, z_trans(z[:, j]).reshape(-1, 1)))
#         elif(feature_type[f] == 'num'):
#             z_1 = np.hstack((z_1, z_trans(z[:, j]).reshape(-1, 1)))
#         else:
#             z_1 = np.hstack((z_1, z[:, j].reshape(-1, 1)))

    np.savez(path, x=X_1, y=y, z=z_1)
    
    print(f"After preprocessing:")
    print(f"X_1.shape = {X_1.shape}")
    print(f"y.shape = {y.shape}")
    print(f"z_1.shape = {z_1.shape}")

    for j in range(X.shape[1]):
        print(task.features[j], len(np.unique(X[:,j])), np.unique(X[:,j]))

    for j in range(z.shape[1]):
        print(task.group[j], len(np.unique(z[:,j])), np.unique(z[:,j], return_counts=True))


# In[60]:


def create_test_split_folder(data_path, ou_folder):
    print(f"data_path = {data_path}, ou_folder = {ou_folder}")
    
    os.system(f"mkdir {ou_folder}; rm -r {ou_folder}/*")
    os.system(f"mkdir {ou_folder}/train")
    os.system(f"mkdir {ou_folder}/valid")
    os.system(f"mkdir {ou_folder}/test")
    
    data = np.load(data_path)
    x, y, z = data['x'], data['y'], data['z']
    assert(x.shape[0] == y.shape[0] and x.shape[0] == z.shape[0])
    n = x.shape[0]
    
    n_split = 5
    n_train = int(n*0.7)
    n_valid = int(n*0.1)
    n_test = n - n_train - n_valid
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    ## test split
    idx_test = idx[:n_test]
    idx_train_valid = idx[n_test:]
    
    for k in range(n_split):
        print(f"split {k}")
        np.random.shuffle(idx_train_valid)
        idx_train = idx_train_valid[:n_train]
        idx_valid = idx_train_valid[n_train:]
        
        train_x, train_y, train_z = x[idx_train], y[idx_train], z[idx_train]
        valid_x, valid_y, valid_z = x[idx_valid], y[idx_valid], z[idx_valid]
        test_x, test_y, test_z = x[idx_test], y[idx_test], z[idx_test]
        
        np.savez(f"{ou_folder}/train/split_{k:03d}", x=train_x, y=train_y, z=train_z)
        np.savez(f"{ou_folder}/valid/split_{k:03d}", x=valid_x, y=valid_y, z=valid_z)
        np.savez(f"{ou_folder}/test/split_{k:03d}", x=test_x, y=test_y, z=test_z)
        


# In[61]:


## self defined prediction task
MyACSIncome = folktables.BasicProblem(
    features=[
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        #'AGEP',
        #'SEX',
        #'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 42000,
    group=['AGEP', 'SEX' ,'RAC1P'],
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

MyACSPublicCoverage = folktables.BasicProblem(
    features=[
        #'AGEP',
        'SCHL',
        'MAR',
        #'SEX',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'PINCP',
        'ESR',
        'ST',
        'FER',
        #'RAC1P',
    ],
    target='PUBCOV',
    target_transform=lambda x: x == 1,
    group=['AGEP', 'SEX' ,'RAC1P'],
    preprocess=public_coverage_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

MyACSMobility = folktables.BasicProblem(
    features=[
        #'AGEP',
        'SCHL',
        'MAR',
        #'SEX',
        'DIS',
        'ESP',
        'CIT',
        'MIL',
        'ANC',
        'NATIVITY',
        'RELP',
        'DEAR',
        'DEYE',
        'DREM',
        #'RAC1P',
        'GCL',
        'COW',
        'ESR',
        'WKHP',
        'JWMNP',
        'PINCP',
    ],
    target="MIG",
    target_transform=lambda x: x == 1,
    group=['AGEP', 'SEX' ,'RAC1P'],
    preprocess=lambda x: x.drop(x.loc[(x['AGEP'] <= 18) | (x['AGEP'] >= 35)].index),
    postprocess=lambda x: np.nan_to_num(x, -1),
)

MyACSEmployment = folktables.BasicProblem(
    features=[
        #'AGEP',
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        #'SEX',
        #'RAC1P',
    ],
    target='ESR',
    target_transform=lambda x: x == 1,
    group=['AGEP', 'SEX' ,'RAC1P'],
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

MyACSTravelTime = folktables.BasicProblem(
    features=[
        #'AGEP',
        'SCHL',
        'MAR',
        #'SEX',
        'DIS',
        'ESP',
        'MIG',
        'RELP',
        #'RAC1P',
        #'PUMA', #!! area code. too many values!
        'ST',
        'CIT',
        'OCCP',
        'JWTR',
        #'POWPUMA', ##!! area code. too many values!
        'POVPIP',
    ],
    target="JWMNP",
    target_transform=lambda x: x > 20,
    group=['AGEP', 'SEX' ,'RAC1P'],
    preprocess=travel_time_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


# In[64]:


## define feature specific transformation
def build_occp_dict(path):
    code_map = {}
    sector_list = []
    with open(path, "r") as fpin:
        for line in fpin:
            tmp = line.strip().split()
            code = tmp[0]
            sector = tmp[1][1:4]
            if(not sector in sector_list):
                sector_list.append(sector)
            code_map[int(code)] = len(sector_list) - 1
            
    ## debug
    print("n_sector = ", len(sector_list), sector_list)
    #print(code_map)
    return code_map

occp_dict = build_occp_dict("./occp_list.txt")
occp_mapper = lambda x : occp_dict[int(x)]

pobp_mapper = lambda x : 0 if x<100 else 1

race_mapper = lambda x : 0 if x<=1 else 1

pincp_mapper = lambda x : np.sign(x)*np.log(np.abs(x)+1.0)

## make a dict of callable mappers
feature_mapper = {"OCCP": occp_mapper, "POBP":pobp_mapper, "RAC1P":race_mapper, "PINCP":pincp_mapper}


# In[65]:


make_data(MyACSIncome, acs_data, "./new_adult_income")
make_data(MyACSPublicCoverage, acs_data, "./new_adult_pubcov")
make_data(MyACSMobility, acs_data, "./new_adult_mobility")
make_data(MyACSEmployment, acs_data, "./new_adult_employment")
make_data(MyACSTravelTime, acs_data, "./new_adult_travelt")


# In[66]:


create_test_split_folder("./new_adult_income.npz", "./new_adult_income/")
create_test_split_folder("./new_adult_pubcov.npz", "./new_adult_pubcov/")
create_test_split_folder("./new_adult_mobility.npz", "./new_adult_mobility/")
create_test_split_folder("./new_adult_employment.npz", "./new_adult_employment/")
create_test_split_folder("./new_adult_travelt.npz", "./new_adult_travelt/")


# In[23]:


## some statistics on income

income_array = acs_data['PINCP'].values
income_array = income_array[~np.isnan(income_array)]

income_mid = np.median(income_array)
income_mean = np.mean(income_array)
print(f"income_mid = {income_mid}, income_mean = {income_mean}")

print("quntile statistics:")
for qi in np.arange(0.0, 1.05, 0.05):
    vi = np.quantile(income_array, qi)
    print(f"{qi*100:4.0f}% ---- {vi:.2f}")

#plt.hist(np.log(np.maximum(income_array, 0.0)+1.0))
logbins = np.logspace(np.log10(1.0),np.log10(np.max(income_array)+1.0),40)
plt.hist(income_array, bins=logbins)
plt.xlim([0, 2e5])
plt.xlabel('income')
plt.ylabel('count')
plt.show()


# ## test

# In[73]:


acs_data['PINCP'].shape


# In[33]:


features, label, group = ACSIncome.df_to_numpy(acs_data)


# In[4]:


print(features.shape)


# In[5]:


features[:5, :]


# In[21]:


acs_data['POBP']


# In[ ]:


data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["FL"], download=True)
features, label, group = ACSIncome.df_to_numpy(acs_data)


# In[ ]:


acs_data.columns


# In[ ]:


features.shape


# In[ ]:


acs_data.shape

