import os
import argparse
import subprocess

import core.environment.env_factory as environment
from core.utils import torch_utils, logger, run_funcs
from core.agent.in_sample import *

file_ids = {'Ant': 'https://storage.googleapis.com/im_datasets/ant.tar', 'Hopper': 'https://storage.googleapis.com/im_datasets/hopper.tar', 'HalfCheetah': 'https://storage.googleapis.com/im_datasets/cheetah.tar', 'Walker2d': 'https://storage.googleapis.com/im_datasets/walker2d.tar'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--env_name', default='Ant', type=str)
    parser.add_argument('--dataset', default='medexp', type=str)
    parser.add_argument('--discrete_control', default=0, type=int)
    parser.add_argument('--state_dim', default=1, type=int)
    parser.add_argument('--action_dim', default=1, type=int)
    parser.add_argument('--tau', default=0.1, type=float)
    
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--log_interval', default=10000, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--timeout', default=1000, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--use_target_network', default=1, type=int)
    parser.add_argument('--target_network_update_freq', default=1, type=int)
    parser.add_argument('--polyak', default=0.995, type=float)
    parser.add_argument('--evaluation_criteria', default='return', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--info', default='0', type=str)
    
    # new arguments for running imbalanced datasets
    parser.add_argument('--dataset_method', default='none', type=str) # or is equal to mixed for using custom datasets
    parser.add_argument('--ratio', default=0.05, type=float)
    parser.add_argument('--dataset_level', default='medium', type=str)

    # regularizer
    parser.add_argument('--lambdaVal', default='none', type=str)


    cfg = parser.parse_args()

    if(cfg.dataset_method != 'none'):
        # command = f'mkdir -p custom_datasets/; cd custom_datasets/; gdown {file_ids[cfg.env_name]} -O {cfg.env_name}.tar.gz; tar -xvzf {cfg.env_name}.tar.gz; rm {cfg.env_name}.tar.gz'
        command = f'mkdir -p custom_datasets/; cd custom_datasets/; wget {file_ids[cfg.env_name]} -O {cfg.env_name}.tar; tar -xvf {cfg.env_name}.tar; rm {cfg.env_name}.tar'
        try:
            # Run the command
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

            # If there is an error, print the error output
            if result.stderr:
                print("Error Output:")
                print(result.stderr)
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    torch_utils.set_one_thread()

    torch_utils.random_seed(cfg.seed)
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    exp_path = "data/output/{}/{}/{}/{}_run".format(cfg.env_name, cfg.dataset, cfg.info, cfg.seed)
    cfg.exp_path = os.path.join(project_root, exp_path)
    torch_utils.ensure_dir(cfg.exp_path)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)
    cfg.offline_data = run_funcs.load_testset(cfg.env_name, cfg.dataset, cfg.seed, cfg.dataset_method, cfg.ratio, cfg.dataset_level)
    # print(cfg.offline_data)
    print("------------------------------------------------Hello------------------------------------------------")

    # Setting up the logger
    # Adding tensorboardX logger
    cfg.tensorboard_logs = True

    cfg.logger = logger.Logger(cfg, cfg.exp_path)
    cfg.device = "cuda"
    logger.log_config(cfg)

    # Initializing the agent and running the experiment
    agent_obj = InSampleAC(
        device=cfg.device,
        discrete_control=cfg.discrete_control,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_units=cfg.hidden_units,
        learning_rate=cfg.learning_rate,
        tau=cfg.tau,
        polyak=cfg.polyak,
        exp_path=cfg.exp_path,
        seed=cfg.seed,
        env_fn=cfg.env_fn,
        timeout=cfg.timeout,
        gamma=cfg.gamma,
        offline_data=cfg.offline_data,
        batch_size=cfg.batch_size,
        use_target_network=cfg.use_target_network,
        target_network_update_freq=cfg.target_network_update_freq,
        evaluation_criteria=cfg.evaluation_criteria,
        logger=cfg.logger,
        lambdaVal = cfg.lambdaVal
    )
    run_funcs.run_steps(agent_obj, cfg.max_steps, cfg.log_interval, exp_path, cfg)