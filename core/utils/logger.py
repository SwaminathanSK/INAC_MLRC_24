import os
import sys

import logging

import wandb

run = 0


def log_config(cfg):
    def get_print_attrs(cfg):
        attrs = dict(cfg.__dict__)
        for k in ['logger', 'env_fn', 'offline_data']:
            del attrs[k]
        return attrs
    attrs = get_print_attrs(cfg)
    for param, value in attrs.items():
        cfg.logger.info('{}: {}'.format(param, value))


class Logger:
    def __init__(self, config, log_dir):
        log_file = os.path.join(log_dir, 'log')
        self._logger = logging.getLogger()
        self.run = None

        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

        self._logger.setLevel(level=logging.INFO)

        self.config = config
        if config.tensorboard_logs:
            self.attrs = dict(config.__dict__)
            for k in ['env_fn', 'offline_data']:
                del self.attrs[k]
            # wandb.login(key = "ab2b40ca778eb2262c7ad70b4887b9a82110f794") # swami
            wandb.login(key = "9693e19323d20b494a26a6ee07f05881b2107bf8") # shreyansh
            print("----------------------------------------------------ATTRIBUTES----------------------------------------------------")
            print("----------------------------------------------------ATTRIBUTES----------------------------------------------------")
            print("----------------------------------------------------ATTRIBUTES----------------------------------------------------")
            print(self.attrs)
            print("----------------------------------------------------ATTRIBUTES----------------------------------------------------")
            print("----------------------------------------------------ATTRIBUTES----------------------------------------------------")
            print("----------------------------------------------------ATTRIBUTES----------------------------------------------------")
            self.run = wandb.init(project="ImbalancedDatasets", config=self.attrs)

    def info(self, log_msg):
        self._logger.info(log_msg)