import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(data):
    # Créer un graphe NetworkX à partir de data
    G = nx.Graph()
    
    # Ajouter les nœuds avec leurs positions
    pos = {i: (data.pos[i][0].item(), data.pos[i][1].item()) for i in range(data.num_nodes)}
    G.add_nodes_from(pos.keys())
    
    # Ajouter les arêtes
    edge_index = data.edge_index.cpu().numpy()
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)

    # Dessiner le graphe
    plt.figure(figsize=(8, 2))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
    plt.title("Visualisation du Graphe 1D")
    plt.show()

def display_graph_info(data):
    # Afficher les informations principales du graphe
    print("=== Paramètres du Graphe ===")
    print(f"data_idx : {data.data_idx}")
    print(f"Nombre de nœuds : {data.num_nodes}")
    print(f"Positions des nœuds :\n{data.pos}")
    print(f"Matrice d'adjacence (edge_index) :\n{data.edge_index}")
    print(f"Informations de grille : {data.grid}")
    print(f"Étapes de message (msg_steps) : {data.msg_steps}")
    print(f"Positions initiales (ini_pos) :\n{getattr(data, 'ini_pos', 'Non défini')}")  # ini_pos si disponible