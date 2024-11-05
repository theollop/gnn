import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def solve_fem_1D(hx, x, f, u0=None, uL=None):
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
    u_torch = torch.tensor(U, dtype=torch.float32)

    return u_torch


def plot_fem_results(fem_solution):
  fem_solution = fem_solution.numpy()

  # Création du gradient de couleur en fonction de u_data
  plt.figure(figsize=(10, 2))
  plt.imshow(fem_solution.reshape(1, -1), cmap='hot', aspect='auto')

  # Ajouter une bordure noire autour du rectangle
  ax = plt.gca()

  plt.colorbar(label="Amplitude de u(x)")  # Affiche la barre de couleur
  plt.axis('off')  # Masque les axes pour une apparence épurée
  plt.title("Gradient de couleur représentant les valeurs de u(x)")
  plt.show()
