import networkx as nx

def generate_ER_network(N, network_dict, seed=None):

    # generate network
    k_mean = network_dict['average_degree']
    p = k_mean / (N - 1)
    G = nx.fast_gnp_random_graph(N, p=p, seed=seed)

    return G

def generate_network(N, Ne, network_dict, seed=None):
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

    if network_dict['network_model'] == 'ER':
        G = generate_ER_network(N, network_dict, seed=seed)
    else:
        raise NotImplementedError(f"The generation of {network_dict['network_model']} network model is not implemented.")

    # get index of agents and split normal agents and experts
    # rename nodes in such a way that nodes 0, 1, ..., Na - 1 are the normal agents and Na, Na + 1, ... N are the experts
    agents = list(G.nodes())
    experts = seed.choice(agents, Ne, replace=False).tolist()
    population = list(set(agents) - set(experts))
    mapping = {node : n_ for n_, node in enumerate(population + experts)}
    G = nx.relabel_nodes(G, mapping)

    # get adjacency matrix
    A = nx.to_scipy_sparse_matrix(G, nodes=agents, format='csr')

    return A