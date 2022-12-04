import json
import time
from collections import defaultdict
import numpy as np
from scipy import sparse
from .utils_network import generate_network
from .utils_model import initialize_model, score_opinion_updates, rmsd_from_truth

MAX_STEPS = 300
CONVERGENCE_THRESHOLD = 1e-3


def update_opinions(Xa, Xe, eps_a, eps_e, alpha_e, tau, A):
    Na = Xa.shape[0]
    X_all = np.hstack([Xa, Xe])
    Xe = Xe[:, np.newaxis]
    eps_all = np.hstack([eps_a, eps_e])[:, np.newaxis]
    eps_e = eps_e[:, np.newaxis]
    alpha_e = alpha_e[:, np.newaxis]

    # TODO:
    # -  is it possible to make it even faster?
    # √ manage divisions by 0 (only Xa can encounter divisions by zero)

    # compute if pairwise differences of opinions under the confidence
    delta_opinions_thresholded = sparse.csr_matrix(np.abs(np.subtract.outer(X_all, X_all)) \
                                                   <= eps_all).astype(int)

    # update Xa
    delta_opinions_thresholded_neigs = delta_opinions_thresholded.multiply(A)
    Xa_new = np.asarray(delta_opinions_thresholded_neigs * X_all[:, np.newaxis]).flatten()
    n_neigs = np.asarray(delta_opinions_thresholded_neigs.sum(axis=1)).flatten()
    Xa_new = np.where(n_neigs == 0, X_all, Xa_new / n_neigs)
    Xa_new = Xa_new[:Na]

    # update Xe
    delta_opinions_thresholded = delta_opinions_thresholded[Na:, Na:]
    Xe_new = np.asarray(delta_opinions_thresholded * Xe / delta_opinions_thresholded.sum(axis=1))
    Xe_new = np.where((np.abs(Xe_new - tau) <= eps_e), # check who is close to the truth
                      alpha_e * tau + (1 - alpha_e) * Xe_new, 
                      Xe_new).flatten()

    return Xa_new, Xe_new


""" Old version, improved runtime
def update_opinions(Xa, Xe, eps_a, eps_e, alpha_e, tau, A):

    Na = Xa.shape[0]
    X_all = np.hstack([Xa, Xe])

    # TODO:
    # - is it possible to make it faster? E.g., matrix formalism?

    # update opinions population
    Xa_new = []
    Xe_new = []
    for i, (Xa_i, eps_a_i) in enumerate(zip(Xa, eps_a)):
        others_within_bound = (np.abs(X_all - Xa_i) <= eps_a_i).astype(int)
        neighbors_within_bound = others_within_bound * A[i].A.flatten()
        #neighbors_within_bound = np.zeros(A.shape[0])
        #neighbors_within_bound =

        n_neigs = neighbors_within_bound.sum()
        xa_new = (neighbors_within_bound * X_all).sum() / n_neigs if n_neigs > 0 else Xa_i
        Xa_new.append(xa_new)

    for i, (Xe_i, eps_e_i, alpha_e_i) in enumerate(zip(Xe, eps_e, alpha_e), Na):
        others_within_bound = (np.abs(Xe - Xe_i) <= eps_e_i).astype(int)
        n_neigs = others_within_bound.sum()
        xe_new = np.abs(others_within_bound * Xe).sum() / n_neigs if n_neigs > 0 else Xe_i
        xe_new = alpha_e_i * tau + (1 - alpha_e_i) * xe_new if np.abs(Xe_i - tau) <= eps_e_i else xe_new
        Xe_new.append(xe_new)

    return np.array(Xa_new), np.array(Xe_new)
"""

#def simulate(N, frac_experts, network_dict, model_param_dict, tau, seed=None, store_history=False):
def simulate(params):
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

    # get params
    N = params['N']
    frac_experts = params['frac_experts']
    network_params = params['network_params']
    init_opinions_params = params['init_opinions_params']
    init_confidence_params = params['init_confidence_params']
    init_alpha_experts_params = params['init_alpha_experts_params']
    tau = params['tau']
    seed = params['seed']
    q = params['q']
    n_simul = params['n_simul']

    t0 = time.time()

    RNG = np.random.RandomState(seed)
    metrics = defaultdict(list)

    # get number of both types of agents
    Ne = int(N * frac_experts)
    Na = N - Ne

    # generate syntetic network
    t0_gen_net = time.time()
    A = generate_network(N, Ne, network_params, seed=RNG)
    t_gen_net = time.time() - t0_gen_net

    # initialize opinions and confidences
    Xa, Xe, eps_a, eps_e, alpha_e = initialize_model(Na, Ne, init_opinions_params, init_confidence_params, init_alpha_experts_params, seed=RNG)

    n_iter = 0
    is_converged = False
    time_updates = 0
    while (n_iter <= MAX_STEPS) and (not is_converged):

        # update opinions
        t0_update = time.time()
        Xa_new, Xe_new = update_opinions(Xa, Xe, eps_a, eps_e, alpha_e, tau, A)
        time_updates += time.time() - t0_update

        # compute metrics and store
        # don't store it every time, store just last one to save space
        #metrics['rmsd_truth_a'].append(rmsd_from_truth(Xa_new, tau))
        #metrics['rmsd_truth_e'].append(rmsd_from_truth(Xe_new, tau))
        #metrics['rmsd_truth_all'].append(rmsd_from_truth(np.hstack([Xa_new, Xe_new]), tau))

        n_iter += 1
        is_converged = score_opinion_updates(np.hstack([Xa, Xe]), np.hstack([Xa_new, Xe_new]), conv_threshold=CONVERGENCE_THRESHOLD)

        Xa = Xa_new
        Xe = Xe_new

    dt = time.time() - t0


    metrics = {**{k:v for k, v in params.items() if k != 'q'}, **dict(metrics)}
    metrics['converged'] = bool(is_converged)
    metrics['total_steps'] = n_iter
    metrics['time_elapsed'] = dt
    metrics['rmsd_truth_a'] = rmsd_from_truth(Xa_new, tau)
    metrics['rmsd_truth_e'] = rmsd_from_truth(Xe_new, tau)
    metrics['rmsd_truth_all'] = rmsd_from_truth(np.hstack([Xa_new, Xe_new]), tau)
    metrics['t_gen_net'] = t_gen_net
    metrics['time_updates'] = time_updates
    

    # store result
    q.put(('write_results', json.dumps(metrics)))

    # log
    log_txt = f'Simulation end in {round(dt/60, 5)} minutes.'
    q.put(('log', log_txt))
