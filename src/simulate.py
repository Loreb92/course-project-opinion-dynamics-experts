from collections import defaultdict
import numpy as np
from utils_network import generate_network
from utils_model import initialize_model, score_opinion_updates, rmsd_from_truth

MAX_STEPS = 1000
CONVERGENCE_THRESHOLD = 1e-4
STORE_ONE_EVERY = 10

def update_opinions(Xa, Xe, eps_a, eps_e, A):

    Na = Xa.shape[0]
    X_all = np.vstack([Xa, Xe])

    # TODO:
    # - is it possible to make it faster? E.g., matrix formalism?

    # update opinions population
    Xa_new = []
    Xe_new = []
    for i, (Xa_i, eps_a_i) in enumerate(zip(Xa, eps_a)):
        others_within_bound = ((X_all - Xa_i).abs() <= eps_a_i).astype(int)
        neighbiors_within_bound = others_within_bound * A[i]
        xa_new = (neighbiors_within_bound * Xa).sum() / neighbiors_within_bound.sum()
        Xa_new.append(xa_new)

    for i, (Xe_i, eps_e_i) in enumerate(zip(Xe, eps_e), Na):
        others_within_bound = ((Xe - Xe_i).abs() <= eps_e_i).astype(int)
        xe_new = (others_within_bound * Xe).abs() / others_within_bound.sum()
        Xe_new.append(xe_new)

    return np.array(Xa_new), np.array(Xe_new)





def simulate(N, frac_experts, network_dict, model_param_dict, tau, seed=None, store_history=False):
    """
    Paramerers:
    N : int, number of agents
    frac_experts : float in [0, 1], fraction of expert agents
    network_dict: dict, contains the information to generate the network
    model_param_dict: dict, contains information on the parameters of the model (e.g., confidence, init opinions)
    tau: float, truth value (in [0, 1])
    seed : int or None, random seed
    store_history : bool, weather to store the history of agents' opinions

    Returns:
    metrics : dict, contains the metrics measured during the simulation
    """

    RNG = np.random.RandomState(seed)
    metrics = defaultdict(list)

    # get number of both types of agents
    Ne = int(N * frac_experts)
    Na = N - Ne

    # generate syntetic network
    A = generate_network(N, Ne, network_dict, seed=seed)

    # initialize opinions and confidences
    Xa, Xe, eps_a, eps_e = initialize_model(Na, Ne, model_param_dict)

    if (store_history):
        metrics['opinions'].append(np.hstack([Xa, Xe]).tolist())

    n_iter = 0
    is_converged = False
    while (n_iter <= MAX_STEPS) and (not is_converged):

        # update opinions
        Xa_new, Xe_new = update_opinions(Xa, Xe, eps_a, eps_e, A)

        # compute metrics and store
        metrics['rmsd_truth_a'].append(rmsd_from_truth(Xa_new, tau))
        metrics['rmsd_truth_e'].append(rmsd_from_truth(Xe_new, tau))
        metrics['rmsd_truth_all'].append(rmsd_from_truth(np.hstack([Xa_new, Xe_new]), tau))

        if (store_history) and ((n_iter + 1) % STORE_ONE_EVERY == 0):
            metrics['opinions'].append(np.hstack([Xa_new, Xe_new]).tolist())

        n_iter += 1
        is_converged = score_opinion_updates(np.hstack([Xa, Xe]), np.hstack([Xa_new, Xe_new]), conv_threshold=CONVERGENCE_THRESHOLD)

        Xa = Xa_new
        Xe = Xe_new

    metrics = dict(metrics)
    metrics['converged'] = is_converged
    metrics['total_steps'] = n_iter

    return metrics