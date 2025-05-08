import os
import numpy as np
from datetime import datetime
import sys
import json
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg_save = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, train_cfg_save, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # save files before training
    # Save dictionary to a JSON file
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir+"/env_config.json", "w") as json_file:
        json.dump(env_cfg_save, json_file)
    with open(log_dir+"/train_config.json", "w") as json_file:
        json.dump(train_cfg_save, json_file)
    print("saved")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
