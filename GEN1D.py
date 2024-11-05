import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt

class GEN1D(nn.Module):
    def __init__(self, num_nodes, device, encoder_dims, decoder_dims):
        super(GEN1D, self).__init__()
        
        # Création de l'encodeur
        self.encoder = nn.ModuleList([
            nn.Linear(encoder_dims[i], encoder_dims[i + 1]) 
            for i in range(len(encoder_dims) - 1)
        ])
        
        # Création du décodeur
        self.decoder = nn.ModuleList([
            nn.Linear(decoder_dims[i], decoder_dims[i + 1]) 
            for i in range(len(decoder_dims) - 1)
        ])
        
        # Création du graphe avec le nombre de noeuds spécifié
        self.G = self.create_graph(num_nodes, device)
        
        # Utilisation de la dernière dimension de l'encodeur comme num_feat pour le graphe
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
        
        # Initialisation des états des noeuds dans le graphe
        self.G.x = X_encoded
        
        # Vérification que edge_index est défini dans le graphe
        assert self.G.edge_index is not None, "edge_index doit être défini dans G pour la convolution graphique"
        
        # Message Passing avec les étapes spécifiées dans msg_steps
        for step in range(self.G.msg_steps):
            self.G.x = self.layer_norm(self.G.x + self.conv(self.G.x, self.G.edge_index))
        
        # Décodage des états finaux des nœuds
        for i, layer in enumerate(self.decoder):
            self.G.x = layer(self.G.x)
            if i + 1 < len(self.decoder):
                self.G.x = F.relu(self.G.x)
        
        X_decoded = self.G.x
        return X_decoded

    def create_graph(self, n, device='cpu', data_idx=-1):
        # Position des noeuds sur une ligne 1D (x, y) avec y = 0
        node_pos = torch.FloatTensor([[i / (n - 1), 0] for i in range(n)]).to(device)
        
        # Initialisation des données du graphe
        data = Data()
        data.data_idx = data_idx
        data.num_nodes = n
        data.pos = node_pos
        data.grid = {'L': 1, 'min_X': 0, 'dx': 1 / (n - 1)}
        data.msg_steps = n - 1  # Nombre d'étapes de message passing
        
        # Création de l'index des arêtes pour une chaîne linéaire de noeuds
        edges = []
        for i in range(n - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        
        return data

    def visualize_graph(self):
        # Créer un graphe NetworkX à partir des données du graphe
        G = nx.Graph()
        
        # Ajouter les nœuds avec leurs positions
        pos = {i: (self.G.pos[i][0].item(), self.G.pos[i][1].item()) for i in range(self.G.num_nodes)}
        G.add_nodes_from(pos.keys())
        
        # Ajouter les arêtes
        edge_index = self.G.edge_index.cpu().numpy()
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)

        # Dessiner le graphe
        plt.figure(figsize=(8, 2))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
        plt.title("Visualisation du Graphe 1D")
        plt.show()

    def display_graph_info(self):
        # Afficher les informations principales du graphe
        print("=== Paramètres du Graphe ===")
        print(f"data_idx : {self.G.data_idx}")
        print(f"Nombre de nœuds : {self.G.num_nodes}")
        print(f"Positions des nœuds :\n{self.G.pos}")
        print(f"Matrice d'adjacence (edge_index) :\n{self.G.edge_index}")
        print(f"Informations de grille : {self.G.grid}")
        print(f"Étapes de message (msg_steps) : {self.G.msg_steps}")
        print(f"Positions initiales (ini_pos) :\n{getattr(self.G, 'ini_pos', 'Non défini')}")  # ini_pos si disponible
