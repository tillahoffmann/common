import numpy as np
import networkx as nx


def erdos_renyi_graph(n, p, seed=None, connected=False):
    """
    Returns a `G_{n,p}` random graph, also known as an Erdos-Renyi graph or
    a binomial graph.

    The `G_{n,p}` model chooses each of the possible edges with probability
    ``p``.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : int, optional
        Seed for random number generator (default=None).
    """
    # Set a seed if desired
    if seed is not None:
        np.random.seed(seed)
    # Iterate until we find a suitable graph
    while True:
        # Create a random binary matrix and construct a graph
        matrix = np.random.uniform(size=(n, n)) < p
        np.fill_diagonal(matrix, 0)
        graph = nx.from_numpy_matrix(matrix)
        # Check whether we need the graph to be connected
        if not connected or nx.is_connected(graph):
            return graph