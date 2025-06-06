# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
from einops import rearrange
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.modules.sub_models.world_models import WorldModel, WorldModel_normal_small_test, WorldModel_normal_small, WorldModel_no_rew_term
from rsl_rl.modules.sub_models.functions_losses import symexp
# from rsl_rl.modules.sub_models.agents import ActorCriticAgent
from rsl_rl.modules.sub_models.replay_buffer import ReplayBuffer, ReplayBuffer_seq, ReplayBuffer_seq_new


def build_world_model(in_channels, action_dim, wm_cfg, privileged_dim):
    # return WorldModel(
    return WorldModel_no_rew_term(
        in_channels=in_channels,
        decoder_out_channels = privileged_dim,
        action_dim=action_dim,
        latent_feature= wm_cfg["latent_feature"],
        transformer_max_length = wm_cfg["max_length"],
        transformer_hidden_dim = wm_cfg["twm_hidden_dim"],
        transformer_num_layers = wm_cfg["twm_num_layers"],
        transformer_num_heads = wm_cfg["twm_num_heads"],
        distribution = wm_cfg["distribution"],
        lr = wm_cfg["learning_rate"],
    ).cuda()

def build_world_model_normal_small(in_channels, wm_cfg, privileged_dim):
    return WorldModel_normal_small(
        in_channels=in_channels,
        decoder_out_channels=privileged_dim,# remove actions (12) and commands (3)
        transformer_max_length = wm_cfg["max_length"],
        transformer_hidden_dim = wm_cfg["twm_hidden_dim"],
        transformer_num_layers = wm_cfg["twm_num_layers"],
        transformer_num_heads = wm_cfg["twm_num_heads"],
        lr = wm_cfg["learning_rate"],
    ).cuda()

def imagine_one_step(worldmodel: WorldModel, sample_obs, sample_action):
    # use the observations sampled from the replay buffer to generate the next observation
    # sample_obs: (batch_size, num_steps, num_envs, num_obs)
    context_latent_flattened = worldmodel.encode_obs(sample_obs) # flattend latent
    for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
        # remember to turn the model to eval mode
        last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = worldmodel.predict_next(
            context_latent_flattened[:, i:i+1],
            sample_action[:, i:i+1],
            log_video=False
        )


class OnPolicy_WM_Runner_train:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.rply_buff = train_cfg["buffer"]
        self.wm_cfg = train_cfg["WM_params"]
        self.wm_training = train_cfg["wm_training"]
        self.img_policy_training = train_cfg["img_policy_training"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            # self.num_critic_obs = self.env.num_privileged_obs 
            self.num_critic_obs = self.env.num_obs
        else:
            self.num_critic_obs = self.env.num_obs
        self.num_critic_obs = self.env.num_obs
        self.num_bool_states = 12

        self.worldmodel = build_world_model_normal_small(self.env.num_privileged_obs - self.num_bool_states, self.wm_cfg, privileged_dim = self.env.num_privileged_obs)
        # self.worldmodel = build_world_model(self.env.num_privileged_obs - self.num_bool_states, self.env.num_actions, self.wm_cfg, privileged_dim = self.env.num_privileged_obs)
        # self.agent = build_agent(self.alg_cfg, self.env.num_actions)

        # build Actor critic class
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        self.num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        load_world_model = False
        load_policy_model = True

        if load_world_model:
            self.worldmodel.load_state_dict(torch.load("/home/aipexws1/conan/unitree_rl_gym/logs/rough_go2_TWM/May08_10-41-46_/world_model_3499.pt"))
            print("loaded pretrained world")
        
        
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        if load_policy_model:
            self.load("/home/aipexws1/conan/unitree_rl_gym/logs/rough_go2/baseline_policy/model_5000.pt")
            self.inference_policy = self.get_inference_policy(device=self.env.device)
            print("loaded pretrained policy")
        # bigger replay buffer for the world model
        self.num_steps_per_env = self.cfg["num_steps_per_env"] # 40, this is also the context length
        self.save_interval = self.cfg["save_interval"] 
        self.replay_buffer = ReplayBuffer_seq_new(
            obs_shape = (self.env.num_obs,),
            priv_obs_shape = (self.env.num_privileged_obs,),
            action_shape=(self.env.num_actions,),
            num_steps_per_env = self.num_steps_per_env,
            num_envs = self.env.num_envs,
            max_length = self.rply_buff["max_len"],
            warmup_length = self.rply_buff["BufferWarmUp"],
            store_on_gpu = self.rply_buff["ReplayBufferOnGPU"],
            device = self.device
        )
        
        

        # world model training parameters
        self.train_dynamics_steps = self.wm_training["wm_train_steps"]
        self.start_train_dynamics_steps = self.wm_training["wm_start_train_steps"]
        self.dreaming_batch_size = self.wm_training["wm_batch_size"]
        self.twm_max_len = self.wm_cfg["max_length"]
        self.train_dynamics_times = self.wm_training["train_wm_times"]
        # init storage and model for the policy
        # self.alg.init_storage_dream(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions], self.dreaming_batch_size)
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [0], [self.env.num_actions])
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_movie_steps = 100
        self.use_imagination = False
        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        # get initial observations
        # obs, privileged_obs, rewards, dones, infos
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        current_obs_loss = 1000
        # dones = self.env.get_dones()
        # infos = self.env.get_extra()
        # critic_obs = privileged_obs if privileged_obs is not None else obs
        critic_obs = obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        ep_infos = []
        collect_steps = 1
        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        """
        The learning algorithm roughly follow the following steps:
        1. Collect num_steps_per_env steps of experience from the environment.
        2. Update policy on environment data
        3. Update world model
        4. updated policy on imagined data generated from world model
        """
        
        for it in range(self.current_learning_iteration, tot_iter):
            data_collected = 0
            start = time.time()
            if it % collect_steps == 0:
                # 1. Collect num_steps_per_env steps of experience from the environment.
                with torch.inference_mode():
                    obs_list = []
                    actions_list = []
                    privilege_obs_list = []
                    reward_list = []
                    dones_list = []
                    term_indexes = []
                    # collect data from the environment
                    for i in range(self.num_steps_per_env):
                        # =============Sample Action=============
                        actions = self.inference_policy(obs.detach())
                        # =============Step the environment=============
                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                        # critic_obs = privileged_obs if privileged_obs is not None else obs
                        critic_obs = obs
                        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        
                        # save the experience to the replay buffer for world model training
                        # there could be different types for saving the experience
                        # aggregated, or not aggregated
                        obs_list.append(obs)
                        privilege_obs_list.append(privileged_obs)
                        actions_list.append(actions)
                        reward_list.append(rewards)
                        dones_list.append(dones)

                    obs_list = torch.stack(obs_list, dim=0)
                    privilege_obs_list = torch.stack(privilege_obs_list, dim=0)
                    actions_list = torch.stack(actions_list, dim=0)
                    reward_list = torch.stack(reward_list, dim=0)
                    dones_list = torch.stack(dones_list, dim=0)

                    obs_list = rearrange(obs_list,"T N F -> N T F")
                    privilege_obs_list = rearrange(privilege_obs_list,"T N F -> N T F")
                    actions_list = rearrange(actions_list,"T N A -> N T A")
                    reward_list = rearrange(reward_list,"T N -> N T")
                    dones_list = rearrange(dones_list,"T N -> N T")

                    # find the indices with termination, we want to exclude them
                    has_true = dones_list.any(dim=1)
                    indices = torch.nonzero(~has_true, as_tuple=True)[0]

                    # filtered the data
                    obs_list = obs_list[indices]
                    privilege_obs_list = privilege_obs_list[indices]
                    actions_list = actions_list[indices]
                    reward_list = reward_list[indices]
                    dones_list = dones_list[indices]

                    self.replay_buffer.append(obs_list, privilege_obs_list, actions_list, reward_list, dones_list)

                    
                # =============end data collection=============
                data_collected = obs_list.shape[0]
            stop = time.time()
            collection_time = stop - start

            if self.replay_buffer.full:
                collect_steps = 500
            start = stop
            # 3. Update world model
            if it%self.train_dynamics_steps == 0 and it > self.start_train_dynamics_steps:
                # 3-1 train tokenizer
                # modify the batch size
                batch_size = self.dreaming_batch_size
                # for it_tok in range(self.train_tokenizer_times):
                #     obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample = self.replay_buffer.sample(batch_size, self.twm_max_len)
                #     # (batch, time, feature)
                #     if it_tok < self.train_tokenizer_times-1:
                #         self.worldmodel.update_tokenizer(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, -1, writer=self.writer)
                #     else:
                #         # only log the final one
                #         self.worldmodel.update_tokenizer(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, it, writer=self.writer)
                
                # 3-2 train dynamics
                for it_wm in range(self.train_dynamics_times):
                    obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample = self.replay_buffer.sample(batch_size, self.twm_max_len, to_device = self.device)
                    if it_wm < self.train_dynamics_times-1:
                        self.worldmodel.update_autoregressive(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, -1, writer=self.writer)
                        # self.worldmodel.update(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, -1, writer=self.writer)
                    else:
                        # only log the final one
                        current_obs_loss = self.worldmodel.update_autoregressive(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, it, writer=self.writer)
                        # self.worldmodel.update(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, it, writer=self.writer)
            stop = time.time()
            learn_WM_time = stop - start
            # =============end updating world model=============            
            print("data collected: ", data_collected)
            print("replay buffer size: ", len(self.replay_buffer))
            print("collection time: ", collection_time)
            print("learn world model time: ", learn_WM_time)

            if it % self.save_interval == 0:    
                torch.save(self.worldmodel.state_dict(), os.path.join(self.log_dir, 'world_model_{}.pt'.format(it)))

            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        torch.save(self.worldmodel.state_dict(), os.path.join(self.log_dir, 'world_model_{}.pt'.format(it)))
        torch.save(self.alg.actor_critic.actor.state_dict(), os.path.join(self.log_dir, 'model_actor_{}.pt'.format(it)))
    
    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
