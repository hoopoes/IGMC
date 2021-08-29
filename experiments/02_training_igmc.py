# IGMC
# 02 Training IGMC

import os, sys
import random
import numpy as np
import pandas as pd

import scipy.sparse as sp
import multiprocessing as mp
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data, Dataset
from torch.optim import Adam
from torch_geometric.data import DataLoader

from tools import *
from utils import *

logger = make_logger(name='igmc_logger')
pd.set_option("display.max_columns", 100)


# 1. Loading
data_dir = os.path.join(os.getcwd(), 'data_dir')

data = load_pickle(os.path.join(data_dir, 'processed_ratings.csv'))
data = data.values

item_features = load_pickle(os.path.join(data_dir, 'processed_item_features.csv'))


# 2. Preprocessing
num_users = np.unique(data[:, 0]).shape[0]
num_items = np.unique(data[:, 1]).shape[0]

logger.info(f"num_users: {num_users}, num_items: {num_items}")

# Train/Val/Test split
data_train_val, data_test = train_test_split(data, test_size=0.2)
data_train, data_val = train_test_split(data_train_val, test_size=0.2)

logger.info(f"Train: {data_train.shape[0]}, Val: {data_val.shape[0]}, Test: {data_test.shape[0]}")

# Create Adjacency Matrix based on train dataset only
# user_item_edges: (4055776, 2) -> idx_nonzero: (4055776,)
adj_mat = sp.dok_matrix(np.zeros(shape=(num_users, num_items)))
for row in data_train:
    adj_mat[row[0], row[1]] = row[2]
adj_mat = adj_mat.tocsr()

# user_item_edges = data_train[:, 0:2]
# user_edges, item_edges = np.split(user_item_edges, indices_or_sections=2, axis=1)
# idx_nonzero: flattened adjacency matrix indices of nonzero values (num_train, )
# (29510, 99) -> (29510 * num_items + 99 = 281702559)
# idx_nonzero = np.array([u * num_items + v for u, v in user_item_edges])

def get_links_and_labels(data):
    links = (data[:, 0], data[:, 1])
    labels = data[:, 2]
    return links, labels

train_links, train_labels = get_links_and_labels(data_train)
val_links, val_labels = get_links_and_labels(data_val)
test_links, test_labels = get_links_and_labels(data_test)

user_features = None
item_features = item_features.values


# 3. Extracting enclosing subgraphs
def create_graphs(A, links, labels, h, max_nodes_per_hop, u_features, v_features):
    graphs = RatingGraphDataset(
        root='data_dir/graphs',
        A=A,
        links=links,
        labels=labels,
        h=h,
        max_nodes_per_hop=max_nodes_per_hop,
        u_features=u_features,
        v_features=v_features
    )
    return graphs


train_graphs = create_graphs(adj_mat, train_links, train_labels, 1, 200, user_features, item_features)
val_graphs = create_graphs(adj_mat, val_links, val_labels, 1, 200, user_features, item_features)
test_graphs = create_graphs(adj_mat, test_links, test_labels, 1, 200, user_features, item_features)


# 해설
# train_graphs.get(0)
# -> Data(edge_index=[2, 7226], edge_type=[7226], x=[297, 4], y=[1])

# np.unique(train_graphs.get(0).edge_index[0].numpy()).shape
# = np.unique(train_graphs.get(0).edge_index[1].numpy()).shape
# user, item 구분 없이 모두 0~296 idx로 표현되는데 이는 torch_geometric의 특징
# 즉, Data.x = (297, 4)로 user, item의 feature 길이가 같아야 함
# 따라서 node feature를 사용할 경우 padding이 필수적임


# 4. Define IGMC
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_adj

from typing import List


class IGMC(nn.Module):
    def __init__(
            self,
            hidden_sizes: List,
            final_emb_size=128,
            num_relations=5,
            num_bases=2,
            adj_dropout=0.2,
            force_undirected=False,
            num_features=4,
            enable_user_features=False,
            enable_item_features=False,
            num_user_features=0,
            num_item_features=0):
        super(IGMC, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.final_emb_size = final_emb_size
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.num_features = num_features
        self.enable_user_features = enable_user_features
        self.enable_item_features = enable_item_features
        self.num_user_features = num_user_features
        self.num_item_features = num_item_features

        num_side_features = num_user_features + num_item_features

        self.convs = torch.nn.ModuleList()

        # RGCN layers
        self.convs.append(
            RGCNConv(in_channels=num_features+num_side_features, out_channels=hidden_sizes[0],
                     num_relations=num_relations, num_bases=num_bases))

        for i in range(0, len(hidden_sizes) - 1):
            self.convs.append(
                RGCNConv(hidden_sizes[i], hidden_sizes[i + 1], num_relations, num_bases))

        self.lin1 = Linear(in_features=2*sum(hidden_sizes)+num_side_features, out_features=final_emb_size)
        self.lin2 = Linear(in_features=final_emb_size, out_features=1)

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, dim=1)

        # data.x에서 user/item 구분하기 (node labeling 기준)
        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], dim=1)
        # 논문에서는 아래와 같이 conv layer를 다 통과하고 나서 side feature를 붙이는데
        # 이 부분은 변형의 여지가 있다. (처음부터 data.x에 다 집어 넣기)
        # 대신 그렇게 하면 위의 RGCN layer를 여러 개 만들어야 할 것이다. (길이가 다르므로)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin2(x)
        # out = out[:, 0]
        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()



model = IGMC(
    hidden_sizes=[32, 32, 32, 32], final_emb_size=128, num_relations=5, num_bases=4,
    adj_dropout=0.2, force_undirected=False, num_features=4
)

total_params = sum(p.numel() for param in model.parameters() for p in param)
print(f'Total number of parameters is {total_params}')


# 5. Train
epochs = 80
batch_size = 50
test_freq = 1
ARR = 0 # 0.001
lr = 1e-3
lr_decay_step_size = 50
lr_decay_factor = 0.1

# Data Loader
num_workers = mp.cpu_count()
train_loader = DataLoader(train_graphs, batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(val_graphs, batch_size, shuffle=False, num_workers=num_workers)

inputs = next(iter(train_loader))


# Pytorch Lightning





