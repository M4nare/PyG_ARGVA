#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import (v_measure_score, homogeneity_score,
                                     completeness_score)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, ARGVA
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot  as plt
import seaborn as sns
import networkx as nx


# In[2]:


from torch_geometric.data import Data
import pandas as pd
import numpy as np


# In[3]:


jpet_edge_index = pd.read_csv('fin_target.csv')
jpet_one_hot = pd.read_csv('weight_jpet.csv',index_col=0)
#jpet_one_hot = pd.read_csv('jpet/test1.csv',index_col=0)


edge_index  = torch.tensor(jpet_edge_index.values,dtype=torch.long)
x = torch.tensor(jpet_one_hot.values, dtype=torch.float)
data = Data(x=x, edge_index=edge_index.t().contiguous())
data


# In[4]:


data.edge_index


# In[5]:


data.x


# In[6]:


data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)
data


# In[7]:


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


# In[8]:


encoder = Encoder(data.num_features, hidden_channels=32, out_channels=32)
discriminator = Discriminator(in_channels=32, hidden_channels=64,
                              out_channels=32)
model = ARGVA(encoder, discriminator)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = model.to(device), data.to(device)

discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                           lr=0.001)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)


# In[9]:


def train():
    model.train()
    encoder_optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)

    for i in range(5):
        discriminator.train()
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z)
        discriminator_loss.backward()
        discriminator_optimizer.step()

    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    encoder_optimizer.step()
    return loss


# In[10]:


@torch.no_grad()
def test():
    G=nx.from_pandas_adjacency(jpet_one_hot, create_using = nx.karate_club_graph())
    G = nx.relabel_nodes(G, { n:str(n) for n in G.nodes()})
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)
    # Cluster embedded values using k-means.
    cluster_input = z.cpu().numpy()
    
    #Kmeans
   
    kmeans = KMeans(n_clusters=4, random_state=0).fit(cluster_input)
    
    #hier
    #ag = AgglomerativeClustering(n_clusters=4)
    #aoc = ag.fit(cluster_input)
    
    #EM
    #gmm = GaussianMixture(n_components=4, random_state=0).fit(cluster_input)
    #gmm_labels = gmm.predict(cluster_input)
    
    for n, label in zip(jpet_one_hot.index.values, kmeans.labels_):
            G.nodes[n]['label'] = label
    
    plt.figure(figsize=(12, 6))
    nx.draw_networkx(G, pos=nx.layout.spring_layout(G), 
                 node_color=[n[1]['label'] for n in G.nodes(data=True)], 
                 cmap=plt.cm.rainbow)
    plt.axis('off')
    
    result_data = np.array([n for n in G.nodes(data=True)])
    
    result_pd = pd.DataFrame(result_data)
    
#    result_pd.to_csv("kmeans_data.csv")
    
    
#    plt.savefig('no_abstract_K.png')
    plt.show()


# In[12]:


for epoch in range(1, 220):
    train()

test()


# In[ ]:




