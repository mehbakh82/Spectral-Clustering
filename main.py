import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import networkx as nx
import graphviz
from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph


class Graph:
    def __init__(self) -> None:
        self.N: int = 0
        self.nodes: set[int] = set()
        self.vertices: dict[int, set[int]] = dict()

    def add_node(self):
        self.nodes.add(self.N)
        self.vertices[self.N] = set()
        self.N += 1

    def connect(self, A: int, B: int):
        assert A < self.N
        assert A >= 0
        assert B < self.N
        assert B >= 0
        assert A != B
        self.vertices[A].add(B)
        self.vertices[B].add(A)

    def disconnect(self, A: int, B: int):
        assert A < self.N
        assert A >= 0
        assert B < self.N
        assert B >= 0
        assert A != B
        assert A in self.vertices[B]
        assert B in self.vertices[A]
        self.vertices[A].remove(B)
        self.vertices[B].remove(A)


def adjacency_matrix(graph: Graph) -> np.ndarray:
    adj_matrix = np.zeros((graph.N, graph.N), dtype=int)
    for node in graph.nodes:
        for neighbor in graph.vertices[node]:
            adj_matrix[node, neighbor] = 1
            adj_matrix[neighbor, node] = 1
    return adj_matrix


def degree_matrix(graph: Graph) -> np.ndarray:
    deg_matrix = np.zeros((graph.N, graph.N), dtype=int)
    for node in graph.nodes:
        deg_matrix[node, node] = len(graph.vertices[node])
    return deg_matrix


def laplacian_matrix(graph: Graph) -> np.ndarray:
    adj_matrix = adjacency_matrix(graph)
    deg_matrix = degree_matrix(graph)
    lap_matrix = deg_matrix - adj_matrix
    return lap_matrix


def laplacian_values(graph: Graph, K: int) -> tuple[np.ndarray, np.ndarray]:
    assert K > 0
    assert K < graph.N
    lap_matrix = laplacian_matrix(graph)
    eigenvalues, eigenvectors = np.linalg.eig(lap_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors


g = Graph()
g.add_node()
g.add_node()
g.add_node()
g.add_node()
eigenvalues, _ = laplacian_values(g, K=g.N - 1)
plt.scatter(range(1, g.N), eigenvalues[1:])
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Laplacian Matrix')
plt.show()

g.connect(0, 1)
eigenvalues, _ = laplacian_values(g, K=g.N - 1)
plt.scatter(range(1, g.N), eigenvalues[1:])
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Laplacian Matrix (After Connecting Nodes 0 and 1)')
plt.show()

g.connect(1, 2)
eigenvalues, _ = laplacian_values(g, K=g.N - 1)
plt.scatter(range(1, g.N), eigenvalues[1:])
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Laplacian Matrix (After Connecting Nodes 1 and 2)')
plt.show()

g.connect(2, 3)
eigenvalues, _ = laplacian_values(g, K=g.N - 1)
plt.scatter(range(1, g.N), eigenvalues[1:])
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Laplacian Matrix (After Connecting Nodes 2 and 3)')
plt.show()

g.connect(3, 0)
eigenvalues, _ = laplacian_values(g, K=g.N - 1)
plt.scatter(range(1, g.N), eigenvalues[1:])
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Laplacian Matrix (After Connecting Nodes 3 and 0)')
plt.show()

g.connect(0, 2)
g.connect(1, 3)
eigenvalues, _ = laplacian_values(g, K=g.N - 1)
plt.scatter(range(1, g.N), eigenvalues[1:])
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Laplacian Matrix (After Connecting Nodes 0-2 and 1-3)')
plt.show()


def cluster(graph: Graph, N: int):
    assert N <= graph.N
    clusterer = KMeans(N)
    _, vecs = laplacian_values(graph, N)
    clusterer.fit(vecs)
    graph.clustered_nodes = dict([i, set()] for i in range(N))
    for N, label in enumerate(clusterer.labels_):
        graph.clustered_nodes[label].add(N)


def plot(graph: Graph):
    G = nx.Graph()
    for A in graph.nodes:
        for B in graph.vertices[A]:
            G.add_edge(A, B)
    pos = nx.spring_layout(G)
    colors = list(mcolors.CSS4_COLORS.values())
    for cluster, nodes in graph.clustered_nodes.items():
        nx.draw_networkx_nodes(G, pos, nodelist=list(nodes), node_color=colors[cluster])
    nx.draw_networkx_edges(G, pos)
    plt.show()


g = Graph()
g.add_node()
g.add_node()
g.connect(0, 1)
g.add_node()
g.connect(0, 2)
g.connect(1, 2)
g.add_node()
g.add_node()
g.connect(3, 4)
g.add_node()
g.connect(3, 5)
g.connect(4, 5)
g.add_node()
g.connect(5, 6)
g.add_node()
g.connect(5, 7)
g.connect(6, 7)
g.add_node()
g.connect(0, 8)
g.add_node()
g.connect(0, 9)
g.connect(8, 9)
g.connect(0, 5)
cluster(g, 3)
plot(g)

X, labels = make_circles(n_samples=2000, noise=.1, factor=.1)
plt.plot(X[:, 0], X[:, 1], 'o')
plt.show()

adj_matrix = kneighbors_graph(X, n_neighbors=5).toarray()
deg_matrix = np.diag(np.sum(adj_matrix, axis=1))
lap_matrix = deg_matrix - adj_matrix


def laplacian_values(laplacian_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors


vals, vecs = laplacian_values(lap_matrix)

clusters = vecs[:, 1] > 0
plt.plot(X[clusters, 0], X[clusters, 1], 'o', label='Cluster 1')
plt.plot(X[~clusters, 0], X[~clusters, 1], 'o', label='Cluster 2')
plt.legend()
plt.show()
