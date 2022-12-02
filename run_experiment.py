import os
import argparse
import json
from itertools import product
from multiprocessing import Pool, Manager

from src.simulate import simulate
from src.logger import Logger

def listener(q, file, logger):
    """
    listens for messages on the q, writes to file.

    Parameters:
    q : Queue object. From its .get() method, it returns a tuple of str (a, b). a refers to the action in ['log', 'write_results'], and b is the string to be written.
    file : the file where to write results
    """

    out_file_f = open(file, 'wt')
    while True:

        m = q.get()
        action, text = m

        if action == 'log':
            logger.write_log(text)
        elif action == 'write_results':
            out_file_f.write(text + '\n')
            out_file_f.flush()
        elif action == 'kill':
            break
        else:
            print(m[0])

    out_file_f.close()


def get_param_list(params):

    keys, values = zip(*params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]

    return permutations_dicts


def run_simulation(params):
    """ 
    Run the simulation and handles eventual errors.
    """
    
    q = params['q']
    
    try:
        simulate(params)
    except KeyboardInterrupt:
        q.put(('log', "Stop for KeyboardInterrupt."))
    except Exception as err:
        import traceback
        traceback = traceback.format_exception(err)
        traceback = "".join(traceback)
        params_show = {k:v for k, v in params.items() if k!='q'}
        q.put(('log', f"Stop collection due to error in data collection. Error: {str(err)}\nParams with error: {json.dumps(params_show)}\n{traceback}"))


def main(args, logger):

    # load params and make it to list of config params
    params = json.load(open(args.input_params_file, 'rt'))
    param_list = get_param_list(params)
    logger.write_log(f"Number of parameters to run: {len(param_list)}")
    param_list = [{**param, 'n_simul':n_simul} for param in param_list for n_simul in range(args.n_runs_per_param)]
    logger.write_log(f"Number of experiments to run: {len(param_list)}")

    # set parallel running of experiments
    # from https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
    manager = Manager()
    q = manager.Queue()
    pool = Pool(args.n_workers)

    # put listener to work first
    watcher = pool.apply_async(listener, (q, args.results_file, logger))

    # fire off workers
    logger.write_log("Start..")
    jobs = []
    seed = args.initial_seed
    for params in param_list:
        params['q'] = q
        params['seed'] = seed
        #job = pool.apply_async(simulate, (params, ))
        job = pool.apply_async(run_simulation, (params, ))
        jobs.append(job)

        seed += 1

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # now we are done, kill the listener
    q.put(('kill', None))
    pool.close()
    pool.join()


if __name__ == '__main__':

    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-params-file', type=str, help='The file (json) containing the parameters of the simulation.')
    parser.add_argument('--results-folder', type=str, help='Folder in which results are stored.')
    parser.add_argument('--n-workers', type=int, default=5, help='Number of processes to run in parallel.')
    parser.add_argument('--log-fold', type=str, default='logs/')
    parser.add_argument('--initial-seed', type=int, default=42, help='The starting seed.')
    parser.add_argument('--n-runs-per-param', type=int, default=10, help='How many times a simulation with a given param config has ot be run.')
    args = parser.parse_args()

    args.exper_name = args.input_params_file.split("/")[-1].replace(".json", "")
    args.results_file = os.path.join(args.results_folder, args.exper_name + '.json')
    args.log_file = os.path.join(args.log_fold, args.exper_name + '.log')

    # check file already exists
    if os.path.exists(args.results_file):
        raise FileExistsError(f"The results file {args.results_file} already exists.")
    if os.path.exists(args.log_file):
        raise FileExistsError(f"The log file {args.log_file} already exists.")

    # SET LOGGER
    logger = Logger(args.log_file)

    main(args, logger)