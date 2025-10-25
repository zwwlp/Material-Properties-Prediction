
import os.path
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from models import GGNNInput

def load_target(target_name, file_path):
    df = pd.read_csv(file_path)
    graph_names = df["name"].values
    targets = df[target_name].values
    return graph_names, targets

def load_graph_data(file_path):
    try:
        graphs = np.load(file_path,allow_pickle=True)['graph_dict'].item()
    except UnicodeError:
        graphs = np.load(file_path, encoding='latin1',allow_pickle=True)['graph_dict'].item()
        graphs = { k.decode() : v for k, v in graphs.items() }
    return graphs

class Graph(object):
    def __init__(self, graph, node_vectors,atom_feature_onehot):
        self.nodes, self.neighbors, self.atom_feature, self.distance_nbr = graph
        self.neighbors = self.neighbors

        n_types = len(node_vectors)
        n_nodes = len(self.nodes)

        # Make node representations
        self.nodes = [node_vectors[i] for i in self.nodes]# map the atomic number to node vector

        self.nodes = np.array(self.nodes, dtype=np.float32)
        if atom_feature_onehot==True:
            self.atom_feature=self.nodes
        else:
            self.atom_feature=self.atom_feature

        self.edge_sources = np.concatenate([[i] * len(self.neighbors[i]) for i in range(n_nodes)])
        self.edge_targets = np.concatenate(self.neighbors)

    def __len__(self):
        return len(self.nodes)

class GraphDataset(Dataset):
    def __init__(self, path, target_name,atom_feature_onehot):
        super(GraphDataset, self).__init__()
        self.atom_feature_onehot=atom_feature_onehot
        self.path = path
        target_path = os.path.join(path, "targets_cutoff.csv")
        self.graph_names, self.targets = load_target(target_name, target_path)
        graph_data_path = os.path.join(path, "graph_data_cutoff.npz")
        self.graph_data = load_graph_data(graph_data_path)
        if atom_feature_onehot==True:
            config_path = os.path.join(path, "config_cutoff.json")
            with open(config_path) as f:
                config = json.load(f)
                self.node_vectors = config["node_vectors"]
        else:
            self.node_vectors=self.graph_data['atom_feature']
        self.graph_data = [Graph(self.graph_data[name], self.node_vectors,atom_feature_onehot)
                           for name in self.graph_names]

    def __getitem__(self, index):
        return self.graph_data[index], self.targets[index], self.graph_names[index]

    def __len__(self):
        return len(self.graph_names)

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_node= [], [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_material_ids = []
    base_idx = 0
    #(atom_fea, nbr_fea, nbr_fea_idx, node_onehot)
    for i, (graph, target,material_id)\
            in enumerate(dataset_list):
        n_i = graph.atom_feature.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(graph.atom_feature)
        batch_nbr_fea.append(graph.distance_nbr)
        batch_nbr_fea_idx.append(np.array(graph.neighbors)+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_material_ids.append(material_id)
        base_idx += n_i
    return (torch.Tensor(np.concatenate(batch_atom_fea, axis=0)),
            torch.Tensor(np.concatenate(batch_nbr_fea, axis=0)),
            torch.LongTensor(np.concatenate(batch_nbr_fea_idx, axis=0)),
            crystal_atom_idx),\
       torch.Tensor(np.stack(batch_target, axis=0)),\
      batch_material_ids

def graph_collate(batch):
    nodes = []
    edge_sources = []
    edge_targets = []
    graph_indices = []
    node_counts = []
    targets = []
    graph_distance_nbr = []
    total_count = 0
    for i, (graph, target,material_id) in enumerate(batch):
        nodes.append(graph.nodes)
        edge_sources.append(graph.edge_sources + total_count)
        edge_targets.append(graph.edge_targets + total_count)
        graph_distance_nbr.append(graph.distance_nbr.reshape(-1,44))
        graph_indices += [i] * len(graph)
        node_counts.append(len(graph))
        targets.append(target)
        total_count += len(graph)

    nodes = np.concatenate(nodes)
    edge_sources = np.concatenate(edge_sources)
    edge_targets = np.concatenate(edge_targets)
    graph_distance_nbr=np.concatenate(graph_distance_nbr)

    input = GGNNInput(nodes, edge_sources, edge_targets,graph_distance_nbr, graph_indices, node_counts)
    targets = torch.Tensor(targets)
    return input, targets

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader
