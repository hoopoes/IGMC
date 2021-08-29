# IGMC
# 01 Preparing Ingredients

# link
# https://www.kaggle.com/CooperUnion/anime-recommendations-database?select=rating.csv

import os, sys
import numpy as np
import pandas as pd

from utils import *

logger = make_logger(name='igmc_logger')
pd.set_option("display.max_columns", 100)


# 1. Load
data_dir = os.path.join(os.getcwd(),'data')

item_meta = pd.read_csv(os.path.join(data_dir, 'item_meta.csv'))
item_meta = item_meta[['anime_id', 'genre']]
item_meta = item_meta.rename({'anime_id': 'item_id'}, axis=1)
item_meta = item_meta.dropna()

# -1, 1~10 -> -1: not rated -> remove
ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
ratings.columns = ['user_id', 'item_id', 'rating']
ratings = ratings[ratings['rating'] != -1]
ratings = ratings[ratings['item_id'].isin(item_meta['item_id'].unique())]

logger.info(f"Num Edges: {ratings.shape[0]}")
logger.info(f"Num Users: {ratings['user_id'].nunique()}")
logger.info(f"Num Items: {ratings['item_id'].nunique()}")

item_meta = item_meta[item_meta['item_id'].isin(ratings['item_id'])]
item_meta = item_meta.reset_index(drop=True)

# Num Edges: 6337239
# Num Users: 69600
# Num Items: 9926


# 2. Preprocess
# genre encoding
genres = set()
for _, row in item_meta[['genre']].iterrows():
    row = {a.strip() for a in row['genre'].split(',')}
    genres = genres.union(row)

item_features = pd.DataFrame(columns=['item_id'] + list(genres))

for index, row in item_meta.iterrows():
    item_features.loc[index, 'item_id'] = row['item_id']

    genre_list = [a.strip() for a in row['genre'].split(',')]
    for genre in genre_list:
        item_features.loc[index, genre] = 1

item_features = item_features.fillna(0)

# Rating 1~10 -> 1~5
map_dict = {i:(i+1)//2 for i in range(1, 11)}

ratings['rating'] = ratings['rating'].map(map_dict)

# sort & reindex id
def map_data(data, col):
    uniq = list(set(data[col]))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data[col] = data[col].map(id_dict)
    data[col] = data[col].astype(np.int64)
    return data

item_features = map_data(item_features, 'item_id')
item_features = item_features.sort_values('item_id')

ratings = map_data(ratings, 'user_id')
ratings = map_data(ratings, 'item_id')

# remove multiple ratings
valid = ratings[['user_id', 'item_id']].drop_duplicates().index.tolist()
ratings = ratings.loc[valid, :]

ratings = ratings.reset_index(drop=True)
item_features = item_features.reset_index(drop=True)

# User ID: 0 ~ 65955 (num_users: 69600)
# Item ID: 0 ~ 9893  (num_item: 9894)

# Save
dump_pickle(os.path.join(data_dir, 'processed_ratings.csv'), ratings)
dump_pickle(os.path.join(data_dir, 'processed_item_features.csv'), item_features)
