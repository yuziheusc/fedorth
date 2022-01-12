# FedOrth: Fair Fedrate Learning with Linear Projection

The package performs fair fedrate learning using linear projection. Features are projected to subspace orthogonal to the sensitive attribute as a preprocess step.
Then fedrate learning model is trained using the preprocessed features. It is ensured that no data is sent from the clients to the server. The projection is also learned in a fedrate manner.

#### Data preparation

Data is stored in .npz format. Data dir contains `n_split` sub dirs. Each dir is the data for a certain split. In each split dir, there are `n_clients` training data files (data/split_001/client_001.npz) for each client and also validation (data/split_001/valid.npz) and (data/split_001/test.npz). For each of the data .npz file, the features are stored in filed 'x', labels are in 'y' and sensitive attributes are in 'z'.

Tools to create splits of data can be found in `flsplits.py`.

#### Create a simple fedrate learning model

```python
fl_model = SyncFl(f"{data_folder}/split_{i:03d}", x_dim=x_dim, n_batch=n_batch, layers=layers, alpha_proj=alpha_proj, decay=decay, mp=False)
fl_model.train(global_epoch, local_epoch)
test_res = fl_model.server.test(fl_model.test_loader)
```

If you want to do experiments on a dataset with multiple splits,

```python
fedproj.run_on_data(data_folder, res_path_i, 103, n_batch=batch_i, layers=layers_i, n_alpha=2, global_epoch=global_epoch_i, local_epoch=local_epoch_i, decay=decay_i)
```