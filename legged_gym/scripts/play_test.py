import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from rsl_rl.modules.sub_models.world_models import WorldModel, WorldModel_normal
import numpy as np
import torch

def build_world_model_normal(in_channels, action_dim, twm_cfg):
    return WorldModel_normal(
        in_channels=in_channels,
        decoder_out_channels=in_channels - 15,# remove actions (12) and commands (3)
        action_dim=action_dim,
        transformer_max_length = twm_cfg.twm_max_len,
        transformer_hidden_dim = twm_cfg.twm_hidden_dim,
        transformer_num_layers = twm_cfg.twm_num_layers,
        transformer_num_heads = twm_cfg.twm_num_heads
    ).cuda()

def build_world_model(in_channels, action_dim, twm_cfg):
    return WorldModel(
        in_channels=in_channels,
        decoder_out_channels= in_channels - 15,
        action_dim=action_dim,
        transformer_max_length = twm_cfg.twm_max_len,
        transformer_hidden_dim = twm_cfg.twm_hidden_dim,
        transformer_num_layers = twm_cfg.twm_num_layers,
        transformer_num_heads = twm_cfg.twm_num_heads
    ).cuda()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    twm_cfg = train_cfg.twm

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # build and load world model
    worldmodel = build_world_model_normal(env.num_obs, env.num_actions,twm_cfg)
    worldmodel.load_state_dict(torch.load("/home/conang/quadruped/unitree_rl_gym/logs/rough_go2_TWM/May07_02-03-59_/world_model_4999.pt"))
    # export policy as a jit module (used to run it from C++)
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # record frames
    # TODO: 1. record the states and actions of the robot interactions with the environment.
    #       2. Use the recorded data (along with the actions) to generate the interactions using world model
    #       3. Use the generated interactions to train the robot using PPO.
    max_episode_length = 200
    
    dreaming_batch_size = 1
    obs_list = []
    actions_list = []
    for i in range(max_episode_length):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        obs_list.append(obs)
        actions_list.append(actions)
    
    
    batch_length = 24
    imagine_horizon = max_episode_length - batch_length
    obs_sample = torch.stack(obs_list, dim=1) # [env_num, episode_length, obs_dim]
    action_sample = torch.stack(actions_list, dim=1) # [env_num, episode_length, action_dim]

    # 
    with torch.inference_mode():
        worldmodel.eval()
        pred_obs, _,_, final_actions = worldmodel.setup_imagination(dreaming_batch_size, obs_sample[:,:batch_length,:], action_sample[:,:batch_length,:], batch_length) 
        cmd_tensor = env.commands[:,:3].squeeze().cuda()
        cmd_tensor = cmd_tensor.repeat(dreaming_batch_size,1,1)
        pred_obs = pred_obs.float()
        pred_obs = torch.cat([pred_obs[:, :, :9],       # First 9 elements of pred_obs
                              cmd_tensor,               # cmd_tensor
                              pred_obs[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
                              final_actions             # final_actions
                             ], dim=-1)
        obs_sample_for_compare = torch.cat((obs_sample[:,:32,:], pred_obs),dim=1)
        for imag_step in range(imagine_horizon):
            # sample action from the policy
            actions = policy(pred_obs[:,-1,:]) # critic_obs is currently the same as obs
            action_sample = torch.cat((action_sample, actions.unsqueeze(dim=1)), dim=1)
            # use the sampled action to do roll outs in the world model
            #=============Step using the world model=============
            pred_obs, critic_obs_sample, reward_sample, termination_sample = worldmodel.imagine_step(_, obs_sample, action_sample, _, _, _)
            pred_obs = torch.cat([pred_obs[:, :, :9],       # First 9 elements of pred_obs
                                    cmd_tensor,               # cmd_tensor
                                    pred_obs[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
                                    actions.unsqueeze(dim=1)  # final_actions
                                    ], dim=-1)
            pred_obs = pred_obs.to(env.device, dtype=torch.float)
            obs_sample_for_compare = torch.cat((obs_sample_for_compare, pred_obs),dim=1)
            rewards, dones = reward_sample.to(env.device,dtype=torch.float).squeeze(), termination_sample.to(env.device,dtype=torch.float).squeeze()
            
    print("obs_sample_for_compare shape: ", obs_sample_for_compare.shape)
    obs_sample_np = obs_sample.squeeze().cpu().numpy()
    obs_hat_np = obs_sample_for_compare.squeeze().cpu().numpy()
    np.save("obs_sample_np", obs_sample_np)
    np.save("obs_hat_np", obs_hat_np)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
