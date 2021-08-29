# Utility Functions

import os, sys
import logging
import pickle
import random
import numpy as np
import scipy.sparse as sp

import torch
from torch_geometric.data import Data, Dataset


def make_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(console)
    return logger

def dump_pickle(address, file):
    with open(address, 'wb') as f:
        pickle.dump(file, f)

def load_pickle(address):
    with open(address, 'rb') as f:
        data = pickle.load(f)
    return data


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, save_path='checkpoint.pt'):
        """
        :param patience: how many times you will wait before earlystopping
        :param save_path: where to save checkpoint
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, val_loss)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, val_loss)
            self.counter = 0 # reset

    def save_checkpoint(self, model, val_loss):
        if self.verbose:
            print(f"val loss: ({self.val_loss_min:.6f} -> {val_loss:.6f})")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


# SparseRowIndexer: A[0:50] 처럼 A의 Row를 기준으로 indexing을 하게 해줌
# SparseColIndexer: A[0:50] 처럼 A의 Col을 기준으로 indexing을 하게 해줌

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return sp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return sp.csc_matrix((data, indices, indptr), shape=shape)


def subgraph_extraction_labeling(
        ind, Arow, Acol, h=1, max_nodes_per_hop=200, u_features=None, v_features=None, ground_truth=5):
    # extract the h-hop enclosing subgraph around link 'ind'
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    for dist in range(1, h + 1):
        # v_fringe = set(Arow[[0]].indices)
        # indices만 따로 저장되어 있으므로 이걸 통해 이웃을 찾으면 full scan을 할 필요가 없다. (sp.mat 특징)
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)

        # user 기준: 방금 찾은 1-hop neighbor인 u_fringe에서 u_visited(자기 자신)을 뺀다.
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited

        # fringe를 정의했으니 지금까지 찾은 걸 모두 visited 집합에 추가함
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)

        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(list(u_fringe), max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(list(v_fringe), max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break

        # [head_node] + u_fringe = 리스트 + 리스트
        # [head_node] + v_fringe
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)

        # dist = 1
        # u_dist = [0, 1, 1,... 1]
        # head: 0, 1-hop neighbors:1
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    # subgraph: sparse adj mat에서 [u_nodes, v_nodes] 기준으로 indexing 한 sub-adj mat
    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0

    # prepare pyg graph constructor input
    # u, v, r = row indices, column indices, values of the nonzero matrix entries.
    u, v, r = sp.find(subgraph)

    # 조정
    v += len(u_nodes)
    r = r - 1  # transform r back to rating label

    # u: 123=len(v_ndoes)개의 item의 1-hop neighbor users 0~18 -> 19명
    # v: 19=len(u_nodes)명의 user의 1-hop neighbor items 19~141 -> 123개
    # subgraph: 19 X 123
    # 논문에 나온 node labeling
    num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    max_node_label = 2 * h + 1

    # get node features
    v_features = v_features[v_nodes]
    node_features = None
    # node_features = [u_features[0], v_features[0]]
    output = [u, v, r, node_labels, max_node_label, ground_truth, node_features]

    return output


def construct_pyg_graph(u, v, r, node_labels, max_node_label, y, node_features):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)
    edge_type = torch.cat([r, r], dim=0)
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    y = torch.FloatTensor([y])
    subgraph = Data(x, edge_index, edge_type=edge_type, y=y)

    if node_features is not None:
        u_feature, v_feature = node_features
        subgraph.u_feature = torch.FloatTensor(u_feature.toarray())
        subgraph.v_feature = torch.FloatTensor(v_feature.toarray())
    return subgraph


# ind = (train_links[0][0], train_links[1][0])
# output = subgraph_extraction_labeling(ind, Arow, Acol, 1, 200, user_features, item_features)
# u, v, r, node_labels, max_node_label, y, node_features = output
# example = construct_pyg_graph(*output)


class RatingGraphDataset(Dataset):
    def __init__(
            self, root, A, links, labels, h, max_nodes_per_hop,
            u_features, v_features
        ):
        """
        :param root: string, where data is stored
        :param A: scipy sparse mat, Full Adjacency Matrix
        :param links: tuple, (user_edge_indices, item_edge_indices)
        :param labels: np.array, ground truth labels
        :param h: int, extract h-hop neighbors
        :param max_nodes_per_hop: int, maximum of number of neighbor nodes per hop
        :param u_features: scipy sparse mat, (num_users, num_user_features)
        :param v_features: scipy sparse mat, (num_items, num_item_features)
        """
        super(RatingGraphDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features

    @property
    def raw_file_names(self):
        return ''

    @property
    def processed_file_names(self):
        return ''

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        y = self.labels[idx]
        output = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, self.h, self.max_nodes_per_hop,
            self.u_features, self.v_features, y
        )
        return construct_pyg_graph(*output)


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x
