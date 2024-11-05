import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

def generate_dataset(hx, x, f, u0=None, uL=None):
    """
    Génère un dataset pour l'équation stationnaire d²u/dx² = f(x) avec des conditions de Dirichlet.

    Paramètres:
        hx (float): Pas de discrétisation.
        x (numpy.ndarray): Positions x discrétisées.
        f (numpy.ndarray): Valeurs de la fonction source f(x).
        u0 (float): Condition de Dirichlet à x = 0 (u(0) = u0).
        uL (float): Condition de Dirichlet à x = L (u(L) = uL).

    Retourne:
        x_torch (torch.Tensor): Positions x discrétisées.
        f_torch (torch.Tensor): Valeurs de la fonction source f(x).
        u_torch (torch.Tensor): Valeurs de la solution numérique u(x).
    """
    N = x.shape[0]
    A = np.zeros((N, N))
    np.fill_diagonal(A, 2 / hx**2)  # Diagonale principale
    np.fill_diagonal(A[1:], -1 / hx**2)  # Diagonale sous-principale
    np.fill_diagonal(A[:, 1:], -1 / hx**2)  # Diagonale sur-principale



    B = np.zeros(N)
    B[1:-1] = f[1:-1]

    # Ajustement pour les conditions aux limites
    if u0 is not None:
        B[0] += u0 / hx**2   # Ajuste B pour inclure la condition u(0) = u0
    if uL is not None:
        B[-1] += uL / hx**2  # Ajuste B pour inclure la condition u(L) = uL

    # Résoudre le système linéaire pour trouver U
    U = np.linalg.solve(A, B)

    # Appliquer les conditions aux limites aux extrémités de U
    if u0 is not None:
        U[0] = u0
    if uL is not None:
        U[-1] = uL


    # Conversion en tenseurs PyTorch
    x_torch = torch.tensor(x, dtype=torch.float32)
    f_torch = torch.tensor(f, dtype=torch.float32)
    u_torch = torch.tensor(U, dtype=torch.float32)

    return x_torch, f_torch, u_torch

# Fonction pour générer un dataset et le convertir en objets Data de PyTorch Geometric
def create_data_object(hx, x, f, u0=None, uL=None):
    x_torch, f_torch, u_torch = generate_dataset(hx, x, f, u0, uL)
    
    # Création de edge_index pour une structure linéaire 1D (connexions entre points voisins)
    edge_index = torch.tensor([[i, i + 1] for i in range(len(x_torch) - 1)] + 
                              [[i + 1, i] for i in range(len(x_torch) - 1)], dtype=torch.long).t()
    
    # Création de l'objet Data
    data = Data(x=f_torch.unsqueeze(1), y=u_torch, edge_index=edge_index)
    return data