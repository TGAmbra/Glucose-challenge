#Import all libraries
import os
import json
import collections
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv

import networkx as nx

#Labels
columns_2018 = ['cbg', 'finger', 'hr', 'gsr', 'carbInput', 'bolus']
columns_2020 = ['cbg', 'finger', 'gsr', 'carbInput', 'bolus']

#get dataframes 
def get_pandas_df(file_path):
  if "2018" in file_path:
    mycolumns = columns_2018
  else: 
    mycolumns = columns_2020

  if "train" in file_path:
    train_file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path,index_col='5minute_intervals_timestamp')
    #count only the labels you care
    final_df = df[mycolumns]
    train_df, val_df = train_test_split(final_df, test_size=0.20, shuffle=False)
    return train_df, val_df
  else:
    test_file_name = os.path.basename(file_path)
    test_df = pd.read_csv(file_path,index_col='5minute_intervals_timestamp')
    test_df = test_df[mycolumns]
    return test_df

# Create adj network
def A_matrix(df):
  # Define the number of temporal neighbors for each time point
  num_temporal_neighbors = 2
  # Initialize an empty adjacency matrix
  x_length = df.shape[0] #number of rows
  adjacency_matrix = np.zeros((x_length, x_length))

  # Populate the adjacency matrix based on temporal neighbors
  for i in range(x_length):
      # Find the indices of the nearest neighbors in terms of temporal proximity
      if x_length == 2:
          # Special case for 2 rows
          neighbors = [0, 1]
      elif 0 < i < x_length-1:
          # If not beginning nor tail, connect to before and after neighbors
          neighbors = [i - 1, i + 1]
      else:
          neighbors = [i - 1]

      # Connect the current time point to its neighbors
      adjacency_matrix[i, neighbors] = 1
      adjacency_matrix[neighbors, i] = 1
  return adjacency_matrix

  # create data obj
def create_data_object(df,adjacency_matrix,columns):
    # Node features
    node_features = df[columns[1:]]  # node0 = cbg, node1 = cbg
    timestamps = df.index
    # Convert to numpy then tensor
    x = torch.tensor(node_features.to_numpy())
    timestamp = torch.tensor(timestamps.to_numpy())
    # Extract labels for each node
    labels = torch.tensor(df[["cbg"]].to_numpy())
    # Create edges
    edges = torch.tensor(np.argwhere(adjacency_matrix!= 0).T, dtype=torch.long)
    # Create Data object
    data = Data(x=x, edge_index=edges.to().contiguous(), y=labels, time=timestamp)
    return data

def create_dataset_objs(train_df, val_df, test_df, train_A, val_A, test_A, columns):
    if train_df.empty:
        print('train_data is empty, please collect more data')
        return None

    if val_df.empty and not (train_df.empty or test_df.empty):
        print('val_data is empty, using the test data')
        # Train data
        train_data = create_data_object(train_df, train_A, columns)
        # Test data
        test_data = create_data_object(test_df, test_A, columns)
        return train_data, test_data

    if test_df.empty and not (train_df.empty or val_df.empty):
        print('test_data is empty, using val_data as test_data')
        # Train data
        train_data = create_data_object(train_df, train_A, columns)
        # Using val_data as test_data
        test_data = create_data_object(val_df, val_A, columns)
        return train_data, test_data

    # Create data objects for train, val, and test, if none is empty
    train_data = create_data_object(train_df, train_A,columns)
    val_data = create_data_object(val_df, val_A,columns)
    test_data = create_data_object(test_df, test_A,columns)
    return train_data, val_data, test_data
