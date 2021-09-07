# IGMC
# 02 Training IGMC

import os, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

import pytorch_lightning as pl

from utils import *
from torch_custom_funcs import *

logger = make_logger(name='igmc_logger')
pd.set_option("display.max_columns", 100)


# 1. Load
data_dir = os.path.join(os.getcwd(), 'data')

data = load_pickle(os.path.join(data_dir, 'processed_ratings.csv'))

num_users = np.unique(data.values[:, 0]).shape[0]
num_items = np.unique(data.values[:, 1]).shape[0]

logger.info(f"num_users: {num_users}, num_items: {num_items}")

sample_ratio = 0.1
data = data.sample(frac=sample_ratio)
data = data.values

# item_features = load_pickle(os.path.join(data_dir, 'processed_item_features.csv'))

user_features = None
item_features = None


# 2. Preprocess
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


train_graphs = create_graphs(adj_mat, train_links, train_labels, 1, 50, user_features, item_features)
val_graphs = create_graphs(adj_mat, val_links, val_labels, 1, 50, user_features, item_features)
#test_graphs = create_graphs(adj_mat, test_links, test_labels, 1, 50, user_features, item_features)

# train_graphs.get(0)
# -> Data(edge_index=[2, 7226], edge_type=[7226], x=[297, 4], y=[1])


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
        # 원본에서는 아래와 같이 conv layer를 다 통과하고 나서 side feature를 붙임
        # 이 부분은 변형의 여지가 있다. (처음부터 data.x에 다 집어 넣기)
        # 대신 그렇게 하면 위의 RGCN layer를 여러 개 만들어야 할 것이다. (길이가 다르므로)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin2(x)
        out = out[:, 0]    # reshaping
        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


# 5. Train
batch_size = 50
ARR = 0 # 0.001
lr_decay_step_size = 50
lr_decay_factor = 0.1

# Data Loader
num_workers = 0    # mp.cpu_count()
train_loader = DataLoader(train_graphs, batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_graphs, batch_size, shuffle=False, num_workers=num_workers)
# inputs = next(iter(train_loader))


# Pytorch Lightning Module
class LightningIGMC(pl.LightningModule):
    def __init__(self, device):
        super().__init__()
        self.model = IGMC(
            hidden_sizes=[32, 32, 32, 32], final_emb_size=128, num_relations=5, num_bases=4,
            adj_dropout=0.2, force_undirected=False, num_features=4).to(device)

    def forward(self, x):
        # forward: defines prediction/inference actions
        score = self.model(x)
        return score

    def training_step(self, batch, batch_idx):
        # training_step = training loop, independent of forward
        # batch: torch_geometric.data.batch.Batch
        y_pred = self.model(batch)
        loss = F.mse_loss(y_pred, batch.y)

        self.log(
            name="train_loss", value=loss,
            prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self.model(batch)
        loss = F.mse_loss(y_pred, batch.y)

        self.log(
            name="val_loss", value=loss,
            prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


device =get_device()
igmc_system = LightningIGMC(device)
total_params = sum(p.numel() for param in igmc_system.model.parameters() for p in param)
logger.info(f"Total number of parameters is {total_params}")

trainer = pl.Trainer(
    gpus=1,
    auto_scale_batch_size="power",
    deterministic=True,
    max_epochs=10
)

trainer.fit(model=igmc_system, train_dataloaders=train_loader, val_dataloaders=val_loader)



