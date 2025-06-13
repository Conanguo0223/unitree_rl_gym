import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from rsl_rl.modules.sub_models.world_models import WorldModel_normal_small, WorldModel, WorldModel_normal
import numpy as np
import torch

def pd_control(actions, q, target_dq, dq, default_dof):
    """Calculates torques from position commands"""
    action_scaled = 0.25*actions
    action_scaled += default_dof
    kp = 20.0
    kd = 0.5
    return (action_scaled - q) * kp + (target_dq - dq) * kd

def build_world_model_normal(in_channels, action_dim, twm_cfg,privileged_dim):
    return WorldModel_normal(
        in_channels=in_channels,
        decoder_out_channels=privileged_dim,# remove actions (12) and commands (3)
        action_dim=action_dim,
        transformer_max_length = twm_cfg.twm_max_len,
        transformer_hidden_dim = twm_cfg.twm_hidden_dim,
        transformer_num_layers = twm_cfg.twm_num_layers,
        transformer_num_heads = twm_cfg.twm_num_heads
    ).cuda()
def build_world_model(in_channels, action_dim, twm_cfg,privileged_dim):
    return WorldModel(
        in_channels=in_channels,
        decoder_out_channels= privileged_dim,
        action_dim=action_dim,
        transformer_max_length = twm_cfg.twm_max_len,
        transformer_hidden_dim = twm_cfg.twm_hidden_dim,
        transformer_num_layers = twm_cfg.twm_num_layers,
        transformer_num_heads = twm_cfg.twm_num_heads
    ).cuda()

def build_world_model_normal_small(in_channels, action_dim, twm_cfg,privileged_dim):
    return WorldModel_normal_small(
        in_channels=in_channels,
        decoder_out_channels=privileged_dim,# remove actions (12) and commands (3)
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
    ppo_runner, train_cfg, train_cfg_dict, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # build and load world model
    worldmodel = build_world_model_normal(env.num_obs, env.num_actions,twm_cfg, privileged_dim = env.num_privileged_obs)
    worldmodel.load_state_dict(torch.load("/home/aipexws1/conan/unitree_rl_gym/logs/rough_go2_TWM_train/May13_01-55-42_/world_model_4999.pt"))
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

    batch_length = twm_cfg.batch_length # context length
    start_episode = 50
    imagine_horizon = max_episode_length - start_episode - batch_length
    obs_sample = torch.stack(obs_list, dim=1) # [env_num, episode_length, obs_dim]
    privilege_obs_sample = torch.stack(privilege_obs_list, dim=1) # [env_num, episode_length, obs_dim]
    action_sample = torch.stack(actions_list, dim=1) # [env_num, episode_length, action_dim]
    reward_sample = torch.stack(reward_list, dim=1) # [env_num, episode_length, action_dim]
    dones_sample = torch.stack(dones_list, dim=1) # [env_num, episode_length, action_dim]
    
    obs_sample_for_compare = privilege_obs_sample[:, :start_episode+batch_length]
    obs_sample_for_inference = obs_sample[:,:start_episode+batch_length,:]
    dreaming_batch_length = 8
    with torch.inference_mode():
        worldmodel.eval()
        # get the cmd_tensor to retrieve the full observation        
        print("command:",env.commands)
        cmd_tensor = env.commands[:,:3].squeeze().cuda() * torch.tensor([2.0,2.0,0.25],device="cuda:0")
        cmd_tensor = cmd_tensor.repeat(dreaming_batch_size,1,1)
        steps = imagine_horizon// dreaming_batch_length
        for imag_step in range(steps):
            # get observations
            obs_inference = obs_sample_for_inference[:,-batch_length:,:]
            action_inference = action_sample[:,start_episode:start_episode+batch_length,:]
            # feed it through the world model
            worldmodel.init_imagine_buffer(dreaming_batch_size, dreaming_batch_length, dtype=worldmodel.tensor_dtype)
            worldmodel.storm_transformer.reset_kv_cache_list(dreaming_batch_size, worldmodel.tensor_dtype)
            # aggregate the context latent
            for i in range(obs_inference.shape[1]):
                last_obs_hat, last_termination_hat, last_latent, last_dist_feat = worldmodel.imagine_step(
                    _, obs_inference[:, i:i+1], action_inference[:,i:i+1], _, _, _
                )
            # H + 1 observation
            action_sample_current = action_sample[:,start_episode+batch_length+dreaming_batch_length*(imag_step)+1:start_episode+batch_length+dreaming_batch_length*(imag_step)+2,:]
            full_obs_for_inf = torch.cat([last_obs_hat[:, :, :9],       # First 9 elements of pred_obs
                                          cmd_tensor,               # cmd_tensor
                                            last_obs_hat[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
                                            action_sample_current
                                            ], dim=-1)
            obs_sample_for_inference = torch.cat((obs_sample_for_inference, full_obs_for_inf),dim=1)
            obs_sample_for_compare = torch.cat((obs_sample_for_compare, last_obs_hat),dim=1)
            for i in range(dreaming_batch_length-1):
                last_obs_hat, critic_obs_sample, reward_sample_img, termination_sample_img = worldmodel.imagine_step(
                    _,
                    full_obs_for_inf,
                    action_sample_current,_,_,_
                )
                action_sample_current = action_sample[:,start_episode+batch_length+dreaming_batch_length*(imag_step)+i+2:start_episode+batch_length+dreaming_batch_length*(imag_step)+3+i,:]
                full_obs_for_inf = torch.cat([last_obs_hat[:, :, :9],       # First 9 elements of pred_obs
                                            cmd_tensor,               # cmd_tensor
                                            last_obs_hat[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
                                            action_sample_current
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
