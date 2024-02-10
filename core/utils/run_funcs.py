import pickle
import h5py
import time
import copy
import numpy as np

import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl
import gzip

EARLYCUTOFF = "EarlyCutOff"


def load_testset(env_name, dataset, id, method, ratio, level):
    path = None
    if env_name == 'HalfCheetah':
        if method == 'mixed':
            if level == 'medium':
                path = {"env": "halfcheetah-medium-v2"}
            elif level == 'expert':
                path = {"env": "halfcheetah-expert-v2"}
        elif dataset == 'expert':
            path = {"env": "halfcheetah-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "halfcheetah-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "halfcheetah-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "halfcheetah-medium-replay-v2"}
    elif env_name == 'Walker2d':
        if method == 'mixed':
            if level == 'medium':
                path = {"env": "walker2d-medium-v2"}
            elif level == 'expert':
                path = {"env": "walker2d-expert-v2"}
        elif dataset == 'expert':
            path = {"env": "walker2d-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "walker2d-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "walker2d-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "walker2d-medium-replay-v2"}
    elif env_name == 'Hopper':
        if method == 'mixed':
            if level == 'medium':
                path = {"env": "hopper-medium-v2"}
            elif level == 'expert':
                path = {"env": "hopper-expert-v2"}
        elif dataset == 'expert':
            path = {"env": "hopper-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "hopper-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "hopper-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "hopper-medium-replay-v2"}
    elif env_name == 'Ant':
        if method == 'mixed':
            if level == 'medium':
                path = {"env": "ant-medium-v2"}
            elif level == 'expert':
                path = {"env": "ant-expert-v2"}
        elif dataset == 'expert':
            path = {"env": "ant-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "ant-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "ant-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "ant-medium-replay-v2"}
    
    elif env_name == 'Acrobot':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/acrobot/expert/ae_run.pkl"}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/acrobot/mixed/am_run.pkl"}
    elif env_name == 'LunarLander':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/lunar_lander/expert/le_run.pkl"}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/lunar_lander/mixed/lm_run.pkl"}
    elif env_name == 'MountainCar':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/mountain_car/expert/me_run.pkl"}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/mountain_car/mixed/mm_run.pkl"}
    elif env_name == 'FourRooms':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/fourrooms/texpert_run.pkl"}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/fourrooms/tmixed_run.pkl"}
        elif dataset == 'missing':
            path = {"pkl": "data/dataset/fourrooms/tmissing_run.pkl"}
        elif dataset == 'random':
            path = {"pkl": "data/dataset/fourrooms/trandom_run.pkl"}
    

    assert path is not None
    testsets = {}
    for name in path:
        if name == "env":
            env = gym.make(path['env'])
            # try:
            if method == 'none':
                data = env.get_dataset()
            elif method == 'mixed':
                file_path = f"custom_datasets/{env_name.lower()}/{env_name.lower()}-random-{level}-{ratio}-v2.hdf5"
                if env_name.lower() == "halfcheetah":
                    file_path = f"custom_datasets/cheetah/{env_name.lower()}-random-{level}-{ratio}-v2.hdf5"

                print(file_path)
                with h5py.File(file_path, 'r') as f:
                    data = {}
                    data['observations'] = list(f['observations'][()])
                    data['actions'] = list(f['actions'][()])
                    data['rewards'] = list(f['rewards'][()])
                    data['next_observations'] = list(f['next_observations'][()])
                    data['terminals'] = list(f['terminals'][()])
            # except:
            #     env = env.unwrapped
            #     data = env.get_dataset()

            testsets[name] = {
                'states': data['observations'],
                'actions': data['actions'],
                'rewards': data['rewards'],
                'next_states': data['next_observations'],
                'terminations': data['terminals'],
            }
        else:
            pth = path[name]
            with open(pth, 'rb') as f:
                dict_temp = pickle.load(f)
                testsets[name] = dict_temp['env']
        return testsets
    else:
        return {}

def run_steps(agent, max_steps, log_interval, eval_pth, config):
    t0 = time.time()
    evaluations = []
    agent.populate_returns(initialize=True)
    while True:
        if log_interval and not agent.total_steps % log_interval:
            mean, median, min_, max_, run_type = agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=True, config=config)
            evaluations.append(mean)
            t0 = time.time()
            # config.logger.run.log({"mean" : mean, "median" : median, "min" : min_, "max" : max_})
            # config.logger.run.log({f"{run_type}_mean" : mean, f"{run_type}_median" : median, f"{run_type}_min" : min_, f"{run_type}_max" : max_, "steps" : log_interval / (time.time() - t0)})

        if max_steps and agent.total_steps >= max_steps:
            break
        agent.step()
    agent.save()
    np.save(eval_pth+"/evaluations.npy", np.array(evaluations))