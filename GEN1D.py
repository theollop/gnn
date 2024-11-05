import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

def create_uniform_1D_graph(n, device='cpu', data_idx=-1):
    node_pos = []

    for i in range(n):
      x = i/(n-1.)
      y = 0
      node_pos.append([x,y])

    node_pos = torch.FloatTensor(np.stack(node_pos, 0)).to(device)
    N = n
    data = Data()
    data.data_idx = data_idx
    data.num_nodes = N
    data.pos = node_pos
    data.grid = {'L': 1, 'min_X' : 0, 'dx':1/(n-1)}
    data.msg_steps = n-1
    # Matrice d'adjacence pour les noeuds
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = data.to(device)
    return data

class GEN1D(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, G=None):
        super(GEN1D, self).__init__()
        
        # Création de l'encodeur
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.encoder.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
        
        # Création du décodeur
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
        
        self.G = G if G is not None else Data()
        
        # Utilisation de la dernière dimension de l'encodeur comme num_feat pour G
        self.G.num_feat = encoder_dims[-1]
        
        # Définition des couches de convolution et de normalisation
        self.conv = GCNConv(self.G.num_feat, self.G.num_feat)
        self.layer_norm = nn.LayerNorm(self.G.num_feat)

    def forward(self, X):
        # Encodage des entrées
        for i, layer in enumerate(self.encoder):
            X = layer(X)
            if i + 1 < len(self.encoder):
                X = F.relu(X)
        X_encoded = X  # X encodé final
        
        # Initialisation des états des noeuds
        self.G.x = X_encoded
        
        # Vérification que edge_index est défini
        assert self.G.edge_index is not None, "edge_index doit être défini dans G pour la convolution graphique"
        
        # Message Passing
        for step in range(self.G.msg_steps):
            self.G.x = self.layer_norm(self.G.x + self.conv(self.G.x, self.G.edge_index))
        
        # Décodage des états finaux des nœuds
        for i, layer in enumerate(self.decoder):
            self.G.x = layer(self.G.x)
            if i + 1 < len(self.decoder):
                self.G.x = F.relu(self.G.x)
        
        X_decoded = self.G.x
        return X_decoded
