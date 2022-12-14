import networkx as nx

def generate_ER_network(N, average_degree, seed=None):

    # generate network
    p = average_degree / (N - 1)
    G = nx.fast_gnp_random_graph(N, p=p, seed=seed)

    return G

def generate_network(N, Ne, network_params, seed=None):
    """
    Generate the network.
    It is not necessary to add the fully connected network between the experts.

    N : int, number of nodes
    Ne : int, number of experts
    network_dict :
    seed : int or np.random.RandomState,

    Returns:
    G : scipy.csr, the adjacency matrix
    """

    if network_params.startswith('ER'):
        _, average_degree = network_params.split("_")
        average_degree = int(average_degree)
        G = generate_ER_network(N, average_degree, seed=seed)
    elif network_params.startswith('empirical'):
        path = network_params.split("_", maxsplit=1)[-1]
        G = nx.read_edgelist(path, delimiter=",", comments='#', nodetype=int)
        G = nx.relabel_nodes(G, {node:i for i, node in enumerate(G.nodes())})
    else:
        raise NotImplementedError(f"The generation of {network_params} network model is not implemented.")

    # get index of agents and split normal agents and experts
    # rename nodes in such a way that nodes 0, 1, ..., Na - 1 are the normal agents and Na, Na + 1, ... N are the experts
    agents = list(G.nodes())
    experts = seed.choice(agents, Ne, replace=False).tolist()
    population = list(set(agents) - set(experts))
    mapping = {node : n_ for n_, node in enumerate(population + experts)}
    G = nx.relabel_nodes(G, mapping)

    # get adjacency matrix
    A = nx.to_scipy_sparse_matrix(G, nodelist=agents, format='csr')

    return A