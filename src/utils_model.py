
import numpy as np

def inizialize_opinions(Na, Ne, init_opinions_dict, seed=None):

    if init_opinions_dict['distrib'] == 'uniform':
        Xa = seed.uniform(0, 1, Na)
        Xe = seed.uniform(0, 1, Ne)
    else:
        raise NotImplementedError("Only unifor initialization of opinions implemented.")

    return Xa, Xe


def initialize_confidence(Na, Ne, init_confidence_dict, seed=None):

    if init_confidence_dict['distrib'] == 'const':
        eps_a, eps_e = np.ones(Na) * init_confidence_dict['confidence'], np.ones(Ne) * init_confidence_dict['confidence']
    else:
        raise NotImplementedError("Only constant confidence implemented.")

    return eps_a, eps_e

def initialize_model(Na, Ne, model_param_dict, seed=None):
    """
    Parameters:
    Na : int, number of population agents
    Ne : int, number of experts
    model_param_dict : dict, contains specifications for the model

    Returns:
    Xa, Xe, eps_a, eps_e : np.array 1D, contains the initial opinions of agents and their confidence
    """

    # initialize opinions and confidence
    Xa, Xe = inizialize_opinions(Na, Ne, model_param_dict['init_opinions'], seed=seed)
    eps_a, eps_e = initialize_confidence(Na, Ne, model_param_dict['init_confidence'], seed=seed)

    # TODO:
    # - eventually, initialize here functions used to update opinions of population and experts and return them

    return Xa, Xe, eps_a, eps_e


def score_opinion_updates(X_old, X_new, conv_threshold):
    return (X_old - X_new).abs().sum() < conv_threshold


def rmsd_from_truth(X, tau):
    return np.sqrt(((X - tau) ** 2).sum() / X.shape[0])