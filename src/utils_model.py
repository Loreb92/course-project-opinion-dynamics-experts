import numpy as np

def inizialize_opinions(Na, Ne, init_opinions_params, seed=None):

    if init_opinions_params == 'uniform':
        Xa = seed.uniform(0, 1, Na)
        Xe = seed.uniform(0, 1, Ne)
    else:
        raise NotImplementedError("Only unifor initialization of opinions implemented.")

    return Xa, Xe


def initialize_confidence(Na, Ne, init_confidence_params, seed=None):

    if init_confidence_params.startswith('const'):
        _, conf = init_confidence_params.split("_")
        conf = float(conf)
        eps_a, eps_e = np.ones(Na) * conf, np.ones(Ne) * conf
    else:
        raise NotImplementedError("Only constant confidence implemented.")

    return eps_a, eps_e


def initialize_alpha_experts(Ne, init_alpha_experts_params, seed=None):

    if init_alpha_experts_params.startswith('const'):
        _, alpha = init_alpha_experts_params.split("_")
        alpha = float(alpha)
        alpha_e = np.ones(Ne) * alpha
    else:
        raise NotImplementedError("Only constant alpha implemented.")

    return alpha_e

def initialize_model(Na, Ne, init_opinions_params, init_confidence_params, init_alpha_experts_params, seed=None):
    """
    Parameters:
    Na : int, number of population agents
    Ne : int, number of experts
    model_param_dict : dict, contains specifications for the model

    Returns:
    Xa, Xe, eps_a, eps_e, alpha_e : np.array 1D, contains the initial opinions of agents and their confidence
    """

    # initialize opinions and confidence, and experts convergence rate
    Xa, Xe = inizialize_opinions(Na, Ne, init_opinions_params, seed=seed)
    eps_a, eps_e = initialize_confidence(Na, Ne, init_confidence_params, seed=seed)
    alpha_e = initialize_alpha_experts(Ne, init_alpha_experts_params, seed=seed)

    # TODO:
    # - eventually, initialize here functions used to update opinions of population and experts and return them

    return Xa, Xe, eps_a, eps_e, alpha_e


def score_opinion_updates(X_old, X_new, conv_threshold):
    return np.abs(X_old - X_new).sum() < conv_threshold


def rmsd_from_truth(X, tau):
    return np.sqrt(((X - tau) ** 2).sum() / X.shape[0])