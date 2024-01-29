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

class simpleGNN(torch.nn.Module):
    def __init__(self,num_of_feat,f):
      #num_of_feat = incoming features, f = input channels
        super(simpleGNN, self).__init__()
        self.conv1 = GCNConv(num_of_feat, f)
        self.conv2 = GCNConv(f, 1)


    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.conv1(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def get_epochs_and_in_feat(train_data):
  num_of_feat=train_data.num_node_features
  if num_of_feat < 31:  # sqr(31)=961
      epochs = num_of_feat ** 2
  elif 31 < num_of_feat < 500:
      epochs = num_of_feat * 2
  else:
      epochs = 1000
  return num_of_feat, epochs

def train(net,train_data,test_data, epochs,val_data=None,lr=0.1):
  optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 00001
  criterion = torch.nn.CrossEntropyLoss()
  best_accuracy=0.0

  global train_losses, train_accuracies
  train_losses=[]
  train_accuracies=[]

  global val_losses,val_accuracies
  val_losses=[]
  val_accuracies=[]

  global test_losses, test_accuracies
  test_losses=[]
  test_accuracies=[]

  for ep in range(epochs+1):
      optimizer.zero_grad()
      train_out=net(train_data)
      train_loss = criterion(train_out, train_data.y)
      train_loss.backward()
      optimizer.step()

      train_losses+=[train_loss]
      train_accuracy=accuracy(train_out, train_data.y)
      train_accuracies+=[train_accuracy]

      #apply to test dataset
      test_out = net(test_data)
      test_loss = criterion(test_out, test_data.y)
      test_losses+=[test_loss]
      test_accuracy = accuracy(test_out, test_data.y)
      test_accuracies+=[test_accuracy]

      if val_data is not None:
            val_out = net(val_data)
            val_loss = criterion(val_out, val_data.y)
            val_losses += [val_loss]
            val_accuracy = accuracy(val_out, val_data.y)
            val_accuracies += [val_accuracy]

            if np.round(val_accuracy, 4) > np.round(best_accuracy, 4):
                print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}, Test_Accuracy: {:.4f}"
                      .format(ep + 1, epochs, train_loss.item(), train_accuracy, val_accuracy, test_accuracy))
                best_accuracy = val_accuracy
      else:
            print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Test_Accuracy: {:.4f}"
                  .format(ep + 1, epochs, train_loss.item(), train_accuracy, test_accuracy))

    #Losses
  train_losses_numpy = [tensor.detach().numpy() for tensor in train_losses]
  test_losses_numpy = [tensor.detach().numpy() for tensor in test_losses]
  plt.plot(train_losses_numpy)
  plt.plot(test_losses_numpy)
  if val_data is not None:
    val_losses_numpy = [tensor.detach().numpy() for tensor in val_losses]
    plt.plot(val_losses_numpy)
  plt.show()

  #Accuracies
  plt.plot(train_accuracies, label='Train Accuracies')
  plt.plot(test_accuracies, label='Test Accuracies')
  if val_data is not None:
    plt.plot(val_accuracies, label='Val Accuracies')
  plt.legend()
  plt.show()

  #save weights and biases
  folder_path = '/content/Checkpoints'
  patient = input("Enter patient number: ") 
  torch.save(net.state_dict(), folder_path+'/'+ patient[:3]+'.pth')

    

