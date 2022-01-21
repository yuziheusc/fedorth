import numpy as np
import matplotlib.pyplot as plt
import pickle

import json
import matplotlib.pyplot as plt
import numpy as np

def list_abs(x):
    return np.abs(np.array(x))

def list_max(x,y):
    return np.maximum(np.array(x), np.array(y))


def avg_res_corr(path, label="None", color=None, marker=None):
    res = pickle.load(open(path, "rb"))
    n = len(res)
    alpha_list = []
    acc_list = []
    corr_list = []
    #print(res[0]['collection'][0]['zs'])
    
    for entry in res:
        acc = np.mean([s["acc"] for s in entry["collection"]])
        corr = np.mean([s["z_corr"] for s in entry["collection"]])
        
        alpha_list.append(entry["alpha"])
        acc_list.append(acc)
        corr_list.append(corr)
        
    plt.scatter(acc_list, corr_list, label=label, marker=marker, color=color)
    
    
def avg_res_disc(path, z_col, label="None", marker=None):
    print(f"path = {path}")
    print(f"z_col = {z_col}")
    
    res = pickle.load(open(path, "rb"))
    n = len(res)
    alpha_list = []
    acc_list = []
    disc_list = []
    
    for entry in res:
        acc = np.mean([s["acc"] for s in entry["collection"]])
        #print(entry["collection"][0].keys())
        
        disc_trial = []
        for s in entry['collection']:
            preds_z0 = s["preds"][s["zs"][:,z_col]<0.5]
            preds_z1 = s["preds"][s["zs"][:,z_col]>=0.5]
            disc_trial.append(np.abs(np.mean(preds_z0) - np.mean(preds_z1)))
            
        alpha_list.append(entry["alpha"])
        acc_list.append(acc)
        disc_list.append(np.mean(disc_trial))
        
        
    plt.scatter(acc_list, disc_list, label=label, marker=marker)


# #code example
# avg_res_corr("../new_adult_income_mlp_h1_single.pkl", 1, label="MLP")
# # ... here, plot multiple data files ...
# # Add your own code for label, legend, title and savefig.
# # For best results, use in jupyter notebooks.
    