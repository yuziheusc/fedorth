#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import glob


# In[2]:


def process_csv(path):
    df = pd.read_csv(path)
    features = [c for c in df if c.startswith('x')]
    x = df[features].values
    y = df['y'].values
    z = df['s'].values
    
    return x, y, z

def client_split(x, y, z, n_client):
    n = len(x)
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_each = int(n/n_client)
    
    xs, ys, zs = [], [], []
    for j in range(n_client):
        j_low = j*n_each
        j_high = j_low + n_each
        if(j==n_client-1):
            j_high = n
        #print(j_low, j_high)
        xs.append(x[j_low:j_high, :])
        ys.append(y[j_low:j_high])
        zs.append(z[j_low:j_high])
        
    return xs, ys, zs

def client_split_random(x, y, z, n_client, buffer, z_balance=None, balance_ratio=0.5):
    assert buffer >=0 and buffer <1.0
    n = len(x)
#     idx = np.arange(n)
#     np.random.shuffle(idx)
        
    z = np.reshape(z, (z.shape[0], -1))
    ## vary sensitive attribute balance
    if(z_balance is not None):
        ## binarize feature if necessary
        f_bin = z[:, z_balance]
        
        #print("unique = ", np.unique(f_bin), np.unique(f_bin)==[0,1])
        
        if(np.unique(f_bin).shape[0] != 2): ## not binary
            f_bin = 1.0*(f_bin>np.median(f_bin)) ## binarize
        
        idx_1, = np.where(f_bin>0.5)
        idx_0, = np.where(f_bin<=0.5)
        n_0 = idx_0.shape[0]
        n_1 = idx_1.shape[0]
        
        print(f"count 0 = {n_0}, count_1 = {n_1}")
        
        ## shuffle rows with respect to balance_ratio
        n_front = x.shape[0]//2
        n_back = x.shape[0] - n_front
        n_1_front = int(n_1*balance_ratio)
        n_1_back = n_1 - n_1_front
        
        assert n_front>=n_1_front
        assert n_back>=n_1_back
        
        idx_front = np.append(idx_1[:n_1_front], idx_0[:n_front-n_1_front])
        idx_back = np.append(idx_1[-n_1_back:], idx_0[-(n_back-n_1_back):])
        
        ## shuffle front and back portion separately
        np.random.shuffle(idx_front)
        np.random.shuffle(idx_back)
        
        idx_balance = np.append(idx_front, idx_back)
        
        #print((np.sort(idx_balance)==np.arange(x.shape[0])).all(), idx_balance.shape)
        
        print(f"n_front = {n_front}, n_back = {n_back}")
        print(f"n_1_front = {n_1_front}, n_1_back = {n_1_back}")
        
        ## apply balanced index to x, y, z
        x = x[idx_balance]
        y = y[idx_balance]
        z = z[idx_balance]
        
        print(f"z_mean_front =  {np.mean(z[:n_front,z_balance])}, z_mean_back = {np.mean(z[-n_back:,z_balance])}")
    
    ## vary client datasize
    
    ## for random size, abandoned method
    if(False):
        pos_sep = [0]
        for i in range(n_client-1):
            cent_i = 1.0/n_client*(i+1)
            left_i = cent_i - 0.5*buffer*1.0/n_client
            right_i = cent_i + 0.5*buffer*1.0/n_client
            sep_i = np.random.uniform(left_i, right_i)
            #print(i, cent_i, left_i, right_i, sep_i)
            pos_sep.append(sep_i)
        pos_sep.append(2.0)
        print("pos_sep = ", pos_sep)
    
    ## vary half of the size. Currently in use.
    ## here buffer is the size portion of data in first n_client/2 clients
    ## buffer = 0.5 is perfect balance
    if(True):
        n_front = n_client//2
        n_back = n_client - n_front
        pos_sep = np.linspace(0, buffer, n_front+1)
        pos_sep = np.append(pos_sep, np.linspace(buffer, 1.0, n_back+1)[1:])
        print("pos_sep = ", pos_sep)
    
    xs, ys, zs = [], [], []
    
    for j in range(n_client):
        j_low = int(n*pos_sep[j])
        j_high = min(int(n*pos_sep[j+1]), n)
        print(f"j_low, j_high, j_count = {j_low}, {j_high}, {j_high-j_low}")
        
        xs.append(x[j_low:j_high, :])
        ys.append(y[j_low:j_high])
        zs.append(z[j_low:j_high])
        
        assert j_high-j_low>=200, f"Too few data in client {j}, n={j_high-j_low}"
        
    ## check that each client gets all values of zs
    if z_balance is not None:
        vz_all = np.unique(z[:, z_balance])
        for i in range(n_client):
            vz_client = np.unique(zs[i][:, z_balance])
            assert len(vz_client) == len(vz_all), "Clients with missing z_balance value!"
    
    
    ##
    n_tot = 0
    for j in range(n_client):
        n_tot += xs[j].shape[0]
    print(f"n_tot_true = {n}, n_tot = {n_tot}")
    
    return xs, ys, zs
        
def process_npz(path):
    npz_data = np.load(path)
    x, y, z = npz_data['x'], npz_data['y'], npz_data['z']
    ##
    return x, y, z


def create_splits_with_train_combine(fdin, fdou, n_client=10, size_buffer=None, fmt='csv',                                    z_balance=None, balance_ratio=-1):
    os.system(f"mkdir {fdou}; rm -r {fdou}/*")
    if(fmt=='csv'):
        train_files = sorted(glob.glob(f"{fdin}/train/*.csv"))
        valid_files = sorted(glob.glob(f"{fdin}/valid/*.csv"))
        test_files = sorted(glob.glob(f"{fdin}/test/*.csv"))
    elif(fmt=='npz'):
        train_files = sorted(glob.glob(f"{fdin}/train/*.npz"))
        valid_files = sorted(glob.glob(f"{fdin}/valid/*.npz"))
        test_files = sorted(glob.glob(f"{fdin}/test/*.npz"))
        
    n_splits = len(train_files)
    
    print(f"n = {n_splits}")
    print(f"train_files = {train_files}")
    print(f"valid_files = {valid_files}")
    print(f"test_files = {test_files}")
    
    ## process train files first since train files need to be combined
    x_train, y_train, z_train = None, None, None
    
    for path in train_files:
        if(fmt=='csv'): 
            x_i, y_i, z_i = process_csv(path)
        elif(fmt=='npz'):
            x_i, y_i, z_i = process_npz(path)
        
        if(x_train is None):
            x_train = x_i; y_train = y_i; z_train = z_i
        else:
            x_train = np.append(x_train, x_i, axis=0)
            y_train = np.append(y_train, y_i, axis=0)
            z_train = np.append(z_train, z_i, axis=0)
    
    x_train = x_train.astype('float32')
    y_train = y_train.astype('int64')
    z_train = z_train.astype('int64')
    
    for i in range(n_splits):
        if(size_buffer is None):
            x_train_list, y_train_list, z_train_list = client_split(x_train, y_train, z_train, n_client=n_client)
        else:
            x_train_list, y_train_list, z_train_list =             client_split_random(x_train, y_train, z_train, n_client=n_client, buffer=size_buffer,
                               z_balance=z_balance, balance_ratio=balance_ratio)
        
        if(fmt=='csv'): 
            x_valid, y_valid, z_valid = process_csv(valid_files[i])
            x_test, y_test, z_test = process_csv(test_files[i])
        elif(fmt=='npz'):
            x_valid, y_valid, z_valid = process_npz(valid_files[i])
            x_test, y_test, z_test = process_npz(test_files[i])
        
        x_valid = x_valid.astype('float32')
        y_valid = y_valid.astype('int64')
        z_valid = z_valid.astype('int64')

        x_test = x_test.astype('float32')
        y_test = y_test.astype('int64')
        z_test = z_test.astype('int64')
        
        fd_i = f"{fdou}/split_{i:03d}"
        os.system(f"mkdir {fd_i}")
        
        for j in range(n_client):
            np.savez(f"{fd_i}/client_{j:03d}", x=x_train_list[j], y=y_train_list[j], z=z_train_list[j])
            
        np.savez(f"{fd_i}/valid", x=x_valid, y=y_valid, z=z_valid)
        np.savez(f"{fd_i}/test", x=x_test, y=y_test, z=z_test)
        
        
        
def create_splits(fdin, fdou, n_client=10, size_buffer=None, fmt='csv',                 z_balance=None, balance_ratio=-1):
    os.system(f"mkdir {fdou}; rm -r {fdou}/*")
    
    if(fmt=='csv'):
        train_files = sorted(glob.glob(f"{fdin}/train/*.csv"))
        valid_files = sorted(glob.glob(f"{fdin}/valid/*.csv"))
        test_files = sorted(glob.glob(f"{fdin}/test/*.csv"))
    elif(fmt=='npz'):
        train_files = sorted(glob.glob(f"{fdin}/train/*.npz"))
        valid_files = sorted(glob.glob(f"{fdin}/valid/*.npz"))
        test_files = sorted(glob.glob(f"{fdin}/test/*.npz"))
        
    n_splits = len(train_files)
    
    print(f"n = {n_splits}")
    print(f"train_files = {train_files}")
    print(f"valid_files = {valid_files}")
    print(f"test_files = {test_files}")
    
    for i in range(n_splits):
        
        if(fmt=='csv'): 
            x_train, y_train, z_train = process_csv(train_files[i])
            x_valid, y_valid, z_valid = process_csv(valid_files[i])
            x_test, y_test, z_test = process_csv(test_files[i])
        elif(fmt=='npz'):
            x_train, y_train, z_train = process_npz(train_files[i])
            x_valid, y_valid, z_valid = process_npz(valid_files[i])
            x_test, y_test, z_test = process_npz(test_files[i])
            
        x_train = x_train.astype('float32')
        y_train = y_train.astype('int64')
        z_train = z_train.astype('float32')        
            
        x_valid = x_valid.astype('float32')
        y_valid = y_valid.astype('int64')
        z_valid = z_valid.astype('float32')

        x_test = x_test.astype('float32')
        y_test = y_test.astype('int64')
        z_test = z_test.astype('float32')
        
        if(size_buffer is None):
            assert z_balance is None, "When size_buffer = None, no balance adjustment allowed!"
            x_train_list, y_train_list, z_train_list = client_split(x_train, y_train, z_train, n_client=n_client)
        else:
            x_train_list, y_train_list, z_train_list =            client_split_random(x_train, y_train, z_train, n_client=n_client, buffer=size_buffer,                               z_balance=z_balance, balance_ratio=balance_ratio)
        
        
        fd_i = f"{fdou}/split_{i:03d}"
        os.system(f"mkdir {fd_i}")
        
        for j in range(n_client):
            np.savez(f"{fd_i}/client_{j:03d}", x=x_train_list[j], y=y_train_list[j], z=z_train_list[j])
            
        np.savez(f"{fd_i}/valid", x=x_valid, y=y_valid, z=z_valid)
        np.savez(f"{fd_i}/test", x=x_test, y=y_test, z=z_test)



