import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from rsl_rl.modules.sub_models.world_models import WorldModel_GRU
import numpy as np
import torch

def pd_control(actions, dof_pos_diff, dof_vel):
    """Calculates torques from position commands"""
    actions_scaled = actions * 0.25
    # dof_pos_diff = -1*obs[9:21]/obs_scales.dof_pos
    # dof_vel = obs[21:33]/obs_scales.dof_vel
    dof_pos_diff = dof_pos_diff / 1.0
    dof_vel = dof_vel / 0.05
    torques = 20*(actions_scaled - dof_pos_diff) - 0.5*dof_vel
    return torques

def build_world_model(in_channels, action_dim, twm_cfg,privileged_dim):
    return WorldModel_GRU(
        in_channels=in_channels,
        decoder_out_channels= privileged_dim,
        action_dim=action_dim,
        transformer_max_length = twm_cfg.twm_max_len,
        transformer_hidden_dim = twm_cfg.twm_hidden_dim,
        transformer_num_layers = twm_cfg.twm_num_layers,
        transformer_num_heads = twm_cfg.twm_num_heads
    ).cuda()

def build_world_model_gru(in_channels, gru_cfg, privileged_dim):
    return WorldModel_GRU(
        obs_dim=in_channels,
        decoder_out_channels=privileged_dim,
        gru_hidden_size=gru_cfg.gru_hidden_size,
        mlp_hidden_size=gru_cfg.mlp_hidden_size,
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
    env_cfg.commands.ranges.lin_vel_x = [0.5,0.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0,0.0]
    env_cfg.commands.ranges.heading = [0.0,0.0]
    env_cfg.env.test = True
    gru_cfg = train_cfg.gru

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, train_cfg_dict, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # build and load world model
    worldmodel = build_world_model_gru(env.num_privileged_obs-12, gru_cfg, privileged_dim = env.num_privileged_obs)
    worldmodel.load_state_dict(torch.load("/home/aipexws1/conan/unitree_rl_gym/logs/rough_go2_GRU_train/May19_17-31-17_/world_model_4999.pt"))
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
    privilege_obs_list = []
    reward_list = []
    dones_list = []
    for i in range(max_episode_length):
        actions = policy(obs.detach())
        obs, privilege_obs, rews, dones, infos = env.step(actions.detach())
        obs_list.append(obs)
        actions_list.append(actions)
        reward_list.append(rews)
        dones_list.append(dones)
        privilege_obs_list.append(privilege_obs)

    batch_length = gru_cfg.batch_length # context length
    start_episode = 50
    imagine_horizon = max_episode_length - start_episode - batch_length
    obs_sample = torch.stack(obs_list, dim=1) # [env_num, episode_length, obs_dim]
    privilege_obs_sample = torch.stack(privilege_obs_list, dim=1) # [env_num, episode_length, obs_dim]
    action_sample = torch.stack(actions_list, dim=1) # [env_num, episode_length, action_dim]
    reward_sample = torch.stack(reward_list, dim=1) # [env_num, episode_length, action_dim]
    dones_sample = torch.stack(dones_list, dim=1) # [env_num, episode_length, action_dim]
    
    obs_sample_for_compare = privilege_obs_sample[:, :start_episode+batch_length]
    obs_sample_for_inference = privilege_obs_sample[:,:start_episode+batch_length,:45]
    dreaming_batch_length = 8
    with torch.inference_mode():
        worldmodel.eval()
        # get the cmd_tensor to retrieve the full observation        
        print("command:",env.commands)
        cmd_tensor = env.commands[:,:3].squeeze().cuda() * torch.tensor([2.0,2.0,0.25],device="cuda:0")
        cmd_tensor = cmd_tensor.repeat(dreaming_batch_size,1,1)
        steps = imagine_horizon// dreaming_batch_length
        for imag_step in range(steps):
            # feed context
            obs_inference = obs_sample_for_inference[:,-batch_length:,:]
            action_inference = action_sample[:,start_episode:start_episode+batch_length,:]
            for i in range(obs_inference.shape[1]):
                if i == 0:
                    last_obs_hat, h = worldmodel(obs_inference[:,i:i+1,:45])
                else:
                    last_obs_hat, h = worldmodel(obs_inference[:,i:i+1,:45], h)
            
            # H + 1 observation
            action_sample_current = action_sample[:,start_episode+batch_length+dreaming_batch_length*(imag_step)+1:start_episode+batch_length+dreaming_batch_length*(imag_step)+2,:]
            torque = pd_control(action_sample_current, last_obs_hat[:, -1, 9:21], last_obs_hat[:, :, 21:33])
            full_obs_for_inf = torch.cat([last_obs_hat[:, :, :33],       # First 9 elements of pred_obs
                                          torque
                                            ], dim=-1)
            obs_sample_for_inference = torch.cat((obs_sample_for_inference, full_obs_for_inf),dim=1)
            obs_sample_for_compare = torch.cat((obs_sample_for_compare, last_obs_hat),dim=1)
            for i in range(dreaming_batch_length-1):
                last_obs_hat, h = worldmodel(obs_inference[:,i:i+1,:45], h)
                action_sample_current = action_sample[:,start_episode+batch_length+dreaming_batch_length*(imag_step)+i+2:start_episode+batch_length+dreaming_batch_length*(imag_step)+3+i,:]
                torque = pd_control(action_sample_current, last_obs_hat[:, -1, 9:21], last_obs_hat[:, :, 21:33])
                full_obs_for_inf = torch.cat([last_obs_hat[:, :, :33],       # First 9 elements of pred_obs
                                              torque
                                            ], dim=-1)
                obs_sample_for_inference = torch.cat((obs_sample_for_inference, full_obs_for_inf),dim=1)
                obs_sample_for_compare = torch.cat((obs_sample_for_compare, last_obs_hat),dim=1)
            
            # imag_step += 8
 
    
    print("obs_sample_for_compare shape: ", obs_sample_for_compare.shape)
    np.save("obs_sample_np", privilege_obs_sample.squeeze().cpu().numpy())
    np.save("obs_hat_np", obs_sample_for_compare.squeeze().cpu().numpy())
    # np.save("rewards", reward_sample.cpu().numpy())
    # np.save("rewards_hat", rewards_for_compare.cpu().numpy())
    # np.save("terminations", dones_sample.cpu().numpy())
    # np.save("terminations_hat", terminations_for_compare.cpu().numpy())
    # x = np.arange(len(obs_sample_np[:,0]))
    # plt.plot(x, obs_sample_np[:,0:3], label='obs_sample')
    # x = np.arange(len(obs_hat_np[:,0]))
    # plt.plot(x, obs_hat_np[:,0:3], label='obs_hat')
    # plt.savefig('test_observations.png')

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
