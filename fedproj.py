#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

import os
import glob
import pickle

#from tqdm.notebook import tnrange, tqdm
from tqdm import tnrange, tqdm
#from alive_progress import alive_bar

from torch.multiprocessing import Process, set_start_method
#set_start_method('spawn')


# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import glob

from sklearn import metrics


# In[3]:


def get_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def get_cov(x, y):
    return np.cov(x, y)[0, 1]

def get_basis_param(X):
    #print(X.shape, X)
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    a = vh.T@(np.diag(s**-1))
    return a

def sign_correction(params):
    #print("params = ", params)
    #print("param_abs = ", param_abs)
    
    param_abs = np.mean(np.abs(params), axis=0)
    
    pivots = np.argmax(param_abs, axis=1)
    #print("pivots = ", pivots)
    
    for i in range(params.shape[0]):
        pivots_val = params[i, pivots, np.arange(params.shape[1])]
        pivots_sign = np.sign(pivots_val)
        #print(pivots_sign)
        params[i,:,:] = params[i,:,:] * pivots_sign
        #print(params[i,:,:])
        
    return params

def estimate_proj_param(Z):
    Z = np.reshape(Z, (Z.shape[0], -1))
    m = Z.shape[1]
    if(m==1):
        return np.ones((1,1))
    
    ## normalize
    Z = Z - Z.mean(axis=0)
    Z = np.nan_to_num(Z / np.sum(Z**2, axis=0)**0.5, nan=0)
    #Z = Z / np.sum(Z**2, axis=0)**0.5
    #Z = Z / (1e-6+np.sum(Z**2, axis=0))**0.5
    #Z = np.nan_to_num(Z, nan=0.)
    
    #Z = Z * np.exp(np.arange(1,m+1)[::-1])# reweighting
    Z = Z * np.arange(1,m+1)[::-1]**4
    
    return get_basis_param(Z)

def ortho_proj(X, Z, A, alpha):
    Z = np.reshape(Z, (Z.shape[0], -1))
    m = Z.shape[1]

    
    Z = Z - Z.mean(axis=0)
    Z = np.nan_to_num(Z / np.sum(Z**2, axis=0)**0.5, nan=0)
    #Z = Z / np.sum(Z**2, axis=0)**0.5
    #Z = Z / (1e-6+np.sum(Z**2, axis=0))**0.5
    #Z = np.nan_to_num(Z, nan=0.)
    
    Z = Z * np.arange(1,m+1)[::-1]**4
    
    U = Z@A
    
    return X - (1.0-alpha)*U@(U.T@X)

def fed_estimate_proj_param(folder):
    assert os.path.exists(folder), "Data folder not found!"


    files = glob.glob(f"{folder}/client_*")
    files = sorted(files)
    #print("files = ", files)
    
    assert len(files)>0, "Empty data folder found!"

    param_list = None
    weight_list = None
    for file in files:
        data = np.load(file)

        assert ('x' in data) and ('y' in data) and ('z' in data), "Data format not known!" 

        x_i = data['x']
        z_i = data['z']
        param_i = estimate_proj_param(z_i)
        weight_i = x_i.shape[0]
    
        if(param_list is None):
            param_list = param_i[np.newaxis,:]
            weight_list = np.array([weight_i])
        else:
            param_list = np.append(param_list, param_i[np.newaxis,:], axis=0)
            weight_list = np.append(weight_list, weight_i)
    
    param_list = sign_correction(param_list)
    
    #print(np.transpose(np.transpose(param_list) * weight_list))
     
    param_avg = (param_list.T*weight_list).T
    param_avg = np.sum(param_avg, axis=0)*1./(np.sum(weight_list))
    return param_avg


# In[4]:


class Net_fc(nn.Module):
    
    known_ou_activation = {"none":lambda x : x,                            "sigmoid":nn.Sigmoid(),                           "softmax":nn.Softmax(dim=1),                           "relu":nn.ReLU(),                           "sin":torch.sin,                          }
    
    def __init__(self, indim, layer_sizes, ou_activation="none", dropout_ratio=None, verbos=False):
        super().__init__()
        
        self.verbos = verbos
        self.indim = indim
        self.layer_sizes = layer_sizes
        
        if(ou_activation not in self.known_ou_activation):
            raise Exception("Unknow output activation function")
        self.ou_activation_fun = self.known_ou_activation[ou_activation]
        
        if(dropout_ratio!=None):
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        
        if(verbos):
            print("Input dim = ", indim)
            print("Layer sizes = ", layer_sizes)
            print("Output activation = ", self.ou_activation_fun)
            print("Dropout = ", self.dropout)
            
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            if(i==0):
                layer_i = nn.Linear(indim, layer_sizes[i])
            else:
                layer_i = nn.Linear(layer_sizes[i-1], layer_sizes[i])
            self.layers.append(layer_i)
            if(verbos):
                print("Layer %d"%(i), layer_i)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.layers)):
            layer_i = self.layers[i]
            x = layer_i(x)
            if(i!=len(self.layers)-1):
                x = F.relu(x)
                if(self.dropout!=None):
                    x = self.dropout(x)
        return self.ou_activation_fun(x)


# In[5]:


class FlServer:
    
    known_loss_fun = {"mse":nn.MSELoss(), "cross_entropy":nn.CrossEntropyLoss()}
    
    def __init__(self, indim, oudim, layer_sizes, ou_activation="none", dropout_ratio=None, loss="mse", ifgpu=True, verbos=False):
        self.model = Net_fc(indim, layer_sizes+[oudim], ou_activation=ou_activation, dropout_ratio=dropout_ratio, verbos=verbos)
        self.param_keys = self.model.state_dict().keys()
        self.loss = self.known_loss_fun[loss]

        self.ifgpu = ifgpu

        if self.ifgpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device("cpu")


        print(f"Server: Device = {self.device}")
        self.model.to(self.device)
        
    def get_central_param(self):
        return self.model.state_dict()
    
    def update_central(self, param):
        assert isinstance(param, dict)
        self.model.load_state_dict(param)
        
    def fed_avg(self, param_list, weight_list):
        param_avg = {}
        weight_tot = np.sum(weight_list)
        
        #print("weight_list", weight_list)
        
        for param, weight in zip(param_list, weight_list):
            assert param.keys() == self.param_keys
            for k in param:
                if not k in param_avg:
                    param_avg[k] = 1./weight_tot*weight*param[k]
                else:
                    param_avg[k] += 1./weight_tot*weight*param[k]
        
        return param_avg
    
    def test(self, loader):
        preds = []
        labels = []
        zs = None
        for i, data in enumerate(loader, 0):
            X, y, z = data   

            X = X.to(self.device)
            y = y.to(self.device)
            z = z.to(self.device)

            y_pred = self.model.forward(X)

            preds = np.append(preds, y_pred.cpu().detach().numpy()[:,1])
            labels = np.append(labels, y.cpu().detach().numpy())
            
            z = z.reshape((z.shape[0], -1))

            if(zs is None):
                zs = z.cpu().detach().numpy()
            else:
                zs = np.vstack((zs, z.cpu().detach().numpy()))

        res = {}
        res["log_loss"] = metrics.log_loss(labels, preds)
        res["acc"] = metrics.accuracy_score(labels, 1*(preds>0.5))
        
        #print(f"zs = {zs}, preds = {np.mean(preds)}, labels_mean = {np.mean(labels)}")
        
        #res["z_corr"] = np.abs(get_corr(preds, zs))
        corr_list = []
        for j in range(zs.shape[1]):
            corr_list.append(np.abs(get_corr(preds, zs[:,j])))
            #corr_list.append(np.abs(get_cov(preds, zs[:,j])))
            
        res["z_corr"] = np.max(corr_list)
        res["z_corr_list"] = corr_list
        
        ## save raw
        res["zs"] = zs
        res["preds"] = preds
        res["labels"] = labels
        
        return res


# In[6]:


class FlClient:

    known_loss_fun = {"mse":nn.MSELoss(), "cross_entropy":nn.CrossEntropyLoss()}
    
    def __init__(self,indim, oudim, layer_sizes, ou_activation="none", dropout_ratio=None, decay=1e-4, loss="mse", ifgpu=True, verbos=False):
        self.model = Net_fc(indim, layer_sizes+[oudim], ou_activation=ou_activation, dropout_ratio=dropout_ratio, verbos=verbos)
        
        self.loss_fun = self.known_loss_fun[loss]
        
        self.weight = 0
        
        
        self.tol=1e-4
        self.max_nochange=10
        
        self.decay = decay

        self.ifgpu = ifgpu

        if self.ifgpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device("cpu")

        print(f"Client: Device = {self.device}")
        self.model.to(self.device)

        #self.n_batch = 8
        #self.n_epoch = 8
    
#     def make_loader(self, X, y):
#         tensor_X = torch.tensor(X)
#         tensor_y = torch.tensor(y)#.to(torch.int64)
        
#         print(tensor_X.dtype, tensor_y.dtype)
        
#         dataset = TensorDataset(tensor_X,tensor_y)
#         dataloader = DataLoader(dataset, batch_size=self.n_batch)
        
#         return dataloader
        
    def get_local_param(self):
#         print("debug : param = ", self.model.state_dict(), self.weight)
        return self.model.state_dict(), self.weight
    
    def set_local_param(self, param):
        self.model.load_state_dict(param)
    
    # def update_local(self, X, y):
    def update_local(self, loader, n_epoch):
        ## debug 
        # print("debug: start update local!")
        
    
        if(isinstance(loader, DataLoader)):
            train_loader = loader
            valid_loader = None
        else:
            train_loader, valid_loader = loader
        
        ## run local training
        optimizer = optim.Adam(self.model.parameters(), weight_decay=self.decay)
        self.weight = 0
        
        loss_list_valid = []
        loss_list_train = []
        
        n_nochange = 0

        for i_epoch in range(n_epoch):
            loss_tmp_train = []
            for i, data in enumerate(train_loader, 0):

                X, y, z = data

                X = X.to(self.device)
                y = y.to(self.device)
                z = z.to(self.device)

                y_pred = self.model.forward(X)
            
                loss_net = self.loss_fun(y_pred, y)
                loss_tot = loss_net

                loss_tot.backward()
                optimizer.step()

                if i_epoch == 0:
                    self.weight += X.shape[0]

                loss_tmp_train.append(loss_net.cpu().detach().numpy())
                
            loss_list_train.append(np.mean(loss_tmp_train))
            
            ## early stopping
            if(valid_loader is not None):
                loss_valid = self.test_local(valid_loader)
                loss_list_valid.append(loss_valid)
                
                if(len(loss_list_valid)>=3):
                    #if(loss_list_valid[-1]-loss_list_valid[-2]<-self.tol*abs(loss_list_valid[-2])):
                    if(loss_list_valid[-1]-np.min(loss_list_valid[:-1])<-self.tol*abs(np.min(loss_list_valid[:-1]))):
                        n_nochange = 0
                    else:
                        n_nochange += 1
                        
                # print(f"loss_list_train = {loss_list_valid}, n_nochange={n_nochange}")
                if(n_nochange > self.max_nochange): 
                    print(f"Early stopping {i_epoch:3d} epoches. ", end = "")
                    break    
                
            else:
                if(len(loss_list_train)>=3):
                    #if(loss_list_train[-1]-loss_list_train[-2]<-self.tol*abs(loss_list_train[-2])):
                    if(loss_list_train[-1]-np.min(loss_list_train[:-1])<-self.tol*abs(np.min(loss_list_train[:-1]))):
                        n_nochange = 0
                    else:
                        n_nochange += 1
                        
                # print(f"loss_list_train = {loss_list_train}, n_nochange={n_nochange}")
                if(n_nochange > self.max_nochange): 
                    print(f"Early stopping {i_epoch:3d} epoches. ", end = "")
                    break

#         ## debug 
#         print("ending param")
#         self.get_local_param()
        
                
    
    def test_local(self, loader):
        
        loss_list = []
        for i, data in enumerate(loader, 0):
            X, y, z = data

            X = X.to(self.device)
            y = y.to(self.device)
            z = z.to(self.device)

            y_pred = self.model.forward(X)

            loss_net = self.loss_fun(y_pred, y).cpu().detach().numpy()
            # loss_tot = loss_net

            loss_list.append(loss_net)
        
        return np.mean(loss_list)
            
        


# In[9]:


## synchrous fl pipeline

class SyncFl:
    
    def read_np_data(self, path):
        data = np.load(path)
        return data
        
    def make_loader(self, path):
        #data = self.read_np_data(path)
        data = self.read_np_data(path)
        
        ## projection
        X, y, z = data['x'], data['y'], data['z']
        X_f = ortho_proj(X, z, self.param_proj, self.alpha_proj)
        
        #print(f"make_loader: X {X.shape}, y {y.shape}, z {z.shape}")
        
        tensor_X = torch.tensor(X_f).float()
        tensor_y = torch.tensor(y)#.to(torch.int64)
        tensor_z = torch.tensor(z)
        
        dataset = TensorDataset(tensor_X, tensor_y, tensor_z)
        dataloader = DataLoader(dataset, batch_size=self.n_batch, shuffle=True)
        
        return dataloader
    
    def make_loader_with_valid(self, path, valid_ratio=None):
        #data = self.read_np_data(path)
        data = self.read_np_data(path)
        
        ## projection
        X, y, z = data['x'], data['y'], data['z']
        X_f = ortho_proj(X, z, self.param_proj, self.alpha_proj)
        
        #print(f"make_loader: X {X.shape}, y {y.shape}, z {z.shape}")
        
        tensor_X = torch.tensor(X_f).float()
        tensor_y = torch.tensor(y)#.to(torch.int64)
        tensor_z = torch.tensor(z)
        
        dataset = TensorDataset(tensor_X, tensor_y, tensor_z)
        if(valid_ratio is None):
            loader = DataLoader(dataset, batch_size=self.n_batch)
            return loader
        else:
            n_valid = int(X.shape[0]*valid_ratio)
            n_train = X.shape[0] - n_valid
            dataset_train, dataset_valid = random_split(dataset, [n_train, n_valid])
            train_loader = DataLoader(dataset_train, batch_size=self.n_batch)
            valid_loader = DataLoader(dataset_valid, batch_size=self.n_batch)
            ## 
            print(f"tot_size = {len(dataset)}, train_size = {len(dataset_train)}, valid_size={len(dataset_valid)},")
            
            return train_loader, valid_loader
            
        
    
    def __init__(self, folder, x_dim, layers=[64], c=1.0, alpha_proj=1.0, decay=1e-4, n_batch=32, mp=False, ifgpu=True):
        self.x_dim = x_dim
        self.layers = layers
        self.c = c
        self.alpha_proj = alpha_proj
        self.n_batch = n_batch
        self.mp = mp
        self.ifgpu = ifgpu
        
        ## estimate fair projection param
        self.param_proj = fed_estimate_proj_param(folder)
        
        path_list = sorted(glob.glob(folder+"/*_*"))
        path_valid = folder + "/valid.npz"
        path_test = folder + "/test.npz"
        
        self.n_client = len(path_list)
        
        print(f"Found {self.n_client} data files:")
        for path in path_list:
            print(f"  {path}")
        print(f"Creating {self.n_client} clients.\n")
        
        ##---- 1. read in all the data, create loader for all of them
        self.loader_list = []
        for path_i in path_list:
            #loader_i = self.make_loader_with_valid(path_i, valid_ratio=0.10)
            loader_i = self.make_loader_with_valid(path_i)
            self.loader_list.append(loader_i)
        
        self.valid_loader = self.make_loader(path_valid)
        self.test_loader = self.make_loader(path_test)
        
        ##---- 2. create server and clients
        self.server = FlServer(self.x_dim, 2, layers, ou_activation="softmax", loss="cross_entropy", ifgpu=self.ifgpu, verbos=False)
        
        self.client_list = []
        for i in range(self.n_client):
            client_i = FlClient(self.x_dim, 2, layers, ou_activation="softmax", loss="cross_entropy", decay=decay, ifgpu=self.ifgpu, verbos=False)
            self.client_list.append(client_i)
            
    def fake_job(self):
        print("debug : fake")
    
    def train(self, server_epoch, client_epoch):
        
        for i_epoch in range(server_epoch):
            print(f"Round {i_epoch}")
            
            ## sending out current central parameter
            param_current = self.server.get_central_param()
            
            ## random sample of clients
            m_sel = max(1, int(self.c*self.n_client))
            sel_client_idx = np.random.choice(np.arange(self.n_client), size=m_sel, replace=False)
            sel_client_idx = sorted(sel_client_idx)
            
            print("  sel_client_idx = ", sel_client_idx)
            
            ## updating each client
            if(self.mp):
                raise NotImplementedError

                # ## folloing code does not work! 
                # process_list = []
                # for idx in sel_client_idx:
                #     print(f"  Updating client {idx} using MP...")
                #     client_i = self.client_list[idx]
                #     loader_i = self.loader_list[idx]
                #     client_i.set_local_param(param_current)
                #     p_i = Process(target=client_i.update_local, args=(loader_i, client_epoch))
                #     process_list.append(p_i)
                #     p_i.start()
                    
                # for p_i in process_list:
                #     p_i.join()
                    
            else:
                for idx in sel_client_idx:
                    print(f"  Updating client {idx}...", end = '')
                    client_i = self.client_list[idx]
                    loader_i = self.loader_list[idx]
                    client_i.set_local_param(param_current)
                    client_i.update_local(loader_i, client_epoch)
                    print("Done.")
            
            ## updating server by federate average
            print(f"Updating server...", end = '')
            
            param_list = []
            weight_list = []
            for idx in sel_client_idx:
                client_i = self.client_list[idx]
                param_i, weight_i = client_i.get_local_param()
                param_list.append(param_i)
                weight_list.append(weight_i)
            #     print("debug : client_i = ", client_i)
            #     print("debug : weight_i = ", client_i.weight)

            # print("debug : weight_list = ", weight_list)

            param_next = self.server.fed_avg(param_list, weight_list)
            self.server.update_central(param_next)
            
            print(f"Done.")
            
            valid_res = self.test(self.valid_loader)
            print("---- valid res ----")
            print(f"  acc = {valid_res['acc']}")
            print(f"  z_corr = {valid_res['z_corr']}")
            print("\n")

    def test(self, loader):
        return self.server.test(loader)
    
    def run_test(self):
        return self.test(self.test_loader)


def run_on_data(data_folder, res_path, x_dim, n_split=5, n_alpha=21, n_batch=16, layers=[64], global_epoch=8, local_epoch=4, decay=1e-4, skip=True):
    print(f"**** Task start ****")
    print(f"  Training on folder [{data_folder}], n_split = {n_split}")
    print(f"  Save to [{res_path}]")
    print(f"  input dim = {x_dim}, n_alpha = {n_alpha}")
    print(f"  n_batch = {n_batch}, layers = {layers}, global_epoch = {global_epoch}, local_epoch = {local_epoch}, decay = {decay}")
    print(f"  skip = {skip}")

    if(skip):
        if(os.path.exists(res_path+"_test.pkl")):
            print(f"**** Skip! ****")
            return

    alpha_list = np.linspace(0.0, 1.0, n_alpha)[::-1]
    
    res_list = []
    valid_res_list = []

    for alpha_proj in tqdm(alpha_list):
        print(f"---- alpha = {alpha_proj} ----")
        
        res_sublist = []
        valid_res_sublist = []

        for i in range(n_split):
            print(f"-- split {i} --")
            param_avg = fed_estimate_proj_param(f"{data_folder}/split_{i:03d}")
            fl_model = SyncFl(f"{data_folder}/split_{i:03d}", x_dim=x_dim, n_batch=n_batch, layers=layers, alpha_proj=alpha_proj, decay=decay, mp=False)
            fl_model.train(global_epoch, local_epoch)
            
            test_res = fl_model.server.test(fl_model.test_loader)
            valid_res = fl_model.server.test(fl_model.valid_loader)

            # print(test_res)
            res_sublist.append(test_res)
            valid_res_sublist.append(valid_res)

        res_entry = {"alpha":alpha_proj, "collection":res_sublist}
        valid_res_entry = {"alpha":alpha_proj, "collection":valid_res_sublist}

        res_list.append(res_entry)
        valid_res_list.append(valid_res_entry)

    pickle.dump(res_list, open(res_path+"_test.pkl", "wb"))    
    pickle.dump(valid_res_list, open(res_path+"_valid.pkl", "wb"))

    print(f"**** Task complete! ****")


# ## example of calling
# if __name__ == "__main__":
#     run_on_data("../../../datasets/fl_compas_trans", "../../compas_mlp_try_script.pkl", 8, n_alpha=5, global_epoch=1, local_epoch=1)


# ## COMPAS

# # In[10]:


# #if __name__ == "__main__":

# if False:

#     data_folder = "../datasets/fl_compas_trans"
#     x_dim = 8

#     n_split = 5
#     alpha_list = np.linspace(0.0, 1.0, 21)
#     res_list = []
#     for alpha_proj in tqdm(alpha_list):
#         print(f"---- alpha = {alpha_proj} ----")
#         res_sublist = []
#         for i in range(n_split):
#             print(f"-- split {i} --")
#             param_avg = fed_estimate_proj_param(f"{data_folder}/split_{i:03d}")
#             fl_model = SyncFl(f"{data_folder}/split_{i:03d}", x_dim=x_dim, alpha_proj=alpha_proj, mp=False)
#             fl_model.train(8, 400)
#             test_res = fl_model.server.test(fl_model.test_loader)
#             # print(test_res)
#             res_sublist.append(test_res)
#         res_entry = {"alpha":alpha_proj, "collection":res_sublist}
#         res_list.append(res_entry)

#     pickle.dump(res_list, open("./compas_mlp.pkl", "wb"))
