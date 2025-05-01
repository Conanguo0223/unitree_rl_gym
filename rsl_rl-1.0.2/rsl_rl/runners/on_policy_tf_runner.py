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
from rsl_rl.modules.sub_models.world_models import WorldModel
from rsl_rl.modules.sub_models.functions_losses import symexp
# from rsl_rl.modules.sub_models.agents import ActorCriticAgent
from rsl_rl.modules.sub_models.replay_buffer import ReplayBuffer, ReplayBuffer_seq


def build_world_model(in_channels, action_dim, twm_cfg):
    return WorldModel(
        in_channels=in_channels,
        action_dim=action_dim,
        transformer_max_length = twm_cfg["twm_max_len"],
        transformer_hidden_dim = twm_cfg["twm_hidden_dim"],
        transformer_num_layers = twm_cfg["twm_num_layers"],
        transformer_num_heads = twm_cfg["twm_num_heads"]
    ).cuda()

# def build_agent(alg_cfg, action_dim):
#     return ActorCriticAgent(
#         feat_dim = 32*32 + alg_cfg["twm_hidden_dim"],
#         num_layers = alg_cfg["Agent"]["num_layers"],
#         hidden_dim = alg_cfg["Agent"]["hidden_dim"],
#         action_dim = action_dim,
#         gamma = alg_cfg["Agent"]["gamma"],
#         lambd = alg_cfg["Agent"]["Lambda"],
#         entropy_coef = alg_cfg["Agent"]["entropyCoef"],
#     ).cuda()


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


class OnPolicy_WM_Runner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.rply_buff = train_cfg["buffer"]
        self.twm_cfg = train_cfg["twm"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            self.num_critic_obs = self.env.num_privileged_obs 
        else:
            self.num_critic_obs = self.env.num_obs
        self.worldmodel = build_world_model(self.env.num_obs, self.env.num_actions, self.twm_cfg)
        # self.agent = build_agent(self.alg_cfg, self.env.num_actions)

        # build Actor critic class
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        self.num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # bigger replay buffer for the world model
        self.num_steps_per_env = self.cfg["num_steps_per_env"] # 24, this is also the context length
        self.imagination_horizon = self.policy_cfg["imagination_horizon"]
        self.save_interval = self.cfg["save_interval"] 
        self.replay_buffer = ReplayBuffer_seq(
            obs_shape = (self.env.num_obs,),
            priv_obs_shape = (self.num_critic_obs,),
            action_shape=(self.env.num_actions,),
            num_steps_per_env = self.num_steps_per_env,
            num_envs = self.env.num_envs,
            max_length = self.rply_buff["max_len"],
            warmup_length = self.rply_buff["BufferWarmUp"],
            store_on_gpu = self.rply_buff["ReplayBufferOnGPU"]
        )
        
        

        # world model training parameters
        self.train_dynamics_steps = self.twm_cfg["twm_train_steps"]
        self.start_train_dynamics_steps = self.twm_cfg["twm_start_train_steps"]
        self.start_train_using_dynamics_steps = self.twm_cfg["twm_start_train_policy_steps"]
        self.train_tw_policy_steps = self.twm_cfg["twm_train_policy_steps"]
        self.dreaming_batch_size = self.twm_cfg["dreaming_batch_size"]
        self.batch_length = self.twm_cfg["batch_length"]
        self.demonstration_batch_size = self.twm_cfg["demonstration_batch_size"]
        self.train_agent_steps = self.twm_cfg["train_agent_steps"]
        self.train_tokenizer_times = self.twm_cfg["train_tokenizer_times"]
        self.train_dynamics_times = self.twm_cfg["train_dynamic_times"]
        # init storage and model for the policy
        self.alg.init_storage_dream(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions], self.dreaming_batch_size)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_movie_steps = 100

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
        
        dones = self.env.get_dones()
        infos = self.env.get_extra()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        # switch the agent to train mode
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100) # calculate mean reward 
        lenbuffer = deque(maxlen=100) # calculate mean episode length
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) # accumulate current reward 
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) # accumulate current episode length

        tot_iter = self.current_learning_iteration + num_learning_iterations
        """
        The learning algorithm roughly follow the following steps:
        1. Collect num_steps_per_env steps of experience from the environment.
        2. Update policy on environment data
        3. Update world model
        4. updated policy on imagined data generated from world model
        """
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # 1. Collect num_steps_per_env steps of experience from the environment.
            with torch.inference_mode():
                obs_buf = []
                critic_obs_buf = []
                actions_buf = []
                reward_buf = []
                dones_buf = []
                for i in range(self.num_steps_per_env):
                    # =============Sample Action=============
                    if self.replay_buffer.ready() and self.twm_cfg["use_context"]:
                        # if replay buffer ready and we want to use context to generate
                        # we can have enough context to make WM predictions
                        # TODO: actor that uses the context of environment length
                        pass
                    else:
                        #if the agent is not using context, and replay is sufficient sample directly from the policy
                        actions = self.alg.act(obs, critic_obs)
                    
                    # =============Step the environment=============
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    # =============Store in replay buffer=============                    
                    # save rewards and dones to the internal buffer for policy update
                    self.alg.process_env_step(rewards, dones, infos)
                    # save the experience to the replay buffer for world model training
                    # there could be different types for saving the experience
                    # aggregated, or not aggregated
                    obs_buf.append(obs)
                    critic_obs_buf.append(critic_obs)
                    actions_buf.append(actions)
                    reward_buf.append(rewards)
                    dones_buf.append(dones)
                    
                    # the aggregated shape of these are 
                    # buf: (num_steps_per_env, num_envs, ...)
                    # during training, transformer should sample from the buffer with samples like
                    # sample: (batch_size, num_steps_per_env, ...)

                    # =============Book keeping=============
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # save the experience to the replay buffer
                obs_buf = torch.stack(obs_buf, dim=0)
                critic_obs_buf = torch.stack(critic_obs_buf, dim=0)
                actions_buf = torch.stack(actions_buf, dim=0)
                reward_buf = torch.stack(reward_buf, dim=0)
                dones_buf = torch.stack(dones_buf, dim=0)

                # rearrange the buffer to be (num_envs, num_steps_per_env, ...)
                obs_buf = rearrange(obs_buf,"T N F -> N T F")
                critic_obs_buf = rearrange(critic_obs_buf,"T N F -> N T F")
                actions_buf = rearrange(actions_buf,"T N A -> N T A")
                reward_buf = rearrange(reward_buf,"T N -> N T")
                dones_buf = rearrange(dones_buf,"T N -> N T")
                self.replay_buffer.append(obs_buf, critic_obs_buf, actions_buf, reward_buf, dones_buf)

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # 2. Update policy on environment data
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            # =============end update policy=============


            start = stop
            # 3. Update world model
            if self.replay_buffer.ready() and it%self.train_dynamics_steps == 0 and it > self.start_train_dynamics_steps:
                # 3-1 train tokenizer
                for it_tok in range(self.train_tokenizer_times):
                    obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample = self.replay_buffer.sample(self.dreaming_batch_size, self.demonstration_batch_size, self.batch_length)
                    # (batch, time, feature)
                    if it_tok < self.train_tokenizer_times-1:
                        self.worldmodel.update_tokenizer(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, -1, writer=self.writer)
                    else:
                        # only log the final one
                        self.worldmodel.update_tokenizer(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, it, writer=self.writer)
                
                # 3-2 train dynamics
                for it_wm in range(self.train_dynamics_times):
                    obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample = self.replay_buffer.sample(self.dreaming_batch_size, self.demonstration_batch_size, self.batch_length)
                    if it_wm < self.train_dynamics_times-1:
                        self.worldmodel.update(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, -1, writer=self.writer)
                    else:
                        # only log the final one
                        self.worldmodel.update(obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample, it, writer=self.writer)
            stop = time.time()
            learn_WM_time = stop - start
            # =============end updating world model=============            

            start = stop
            # 4. update policy on imagined data generated from world model
            if self.replay_buffer.ready() and it%self.train_tw_policy_steps == 0 and it > self.start_train_using_dynamics_steps:
                # 4-1 imagine data
                for it_im in range(self.train_agent_steps):
                    with torch.inference_mode():
                        self.worldmodel.eval() # switch to evaluation mode (dropout for example)
                        # sample actual experience from replay buffer
                        obs_sample, critic_obs_sample, action_sample, reward_sample, termination_sample = self.replay_buffer.sample(self.dreaming_batch_size, self.demonstration_batch_size, self.batch_length)
                        # setup the context length of the model using the sampled obs ... (24 steps)
                        # similar like the prompt in transformer models, we need to setup the context, and pre calculate the KV-cache
                        # this should generate the next predicted output using the sampled action.
                        pred_obs, _,_ = self.worldmodel.setup_imagination(self.dreaming_batch_size, obs_sample, action_sample,self.batch_length) 
                        pred_obs = pred_obs.float()
                        pred_critic_obs = privileged_obs if privileged_obs is not None else pred_obs
                        obs_sample_for_inference = torch.cat((obs_sample, pred_obs),dim=1)
                        # rollout the world model
                        for imag_step in range(self.imagination_horizon):
                            # sample action from the policy
                            # TODO: should just use the final state of the observation to get the action and apply it.
                            actions = self.alg.act(pred_obs[:,-1,:], pred_critic_obs[:,-1,:]) # critic_obs is currently the same as obs
                            action_sample = torch.cat((action_sample, actions.unsqueeze(dim=1)), dim=1)
                            # use the sampled action to do roll outs in the world model
                            #=============Step using the world model=============
                            pred_obs, critic_obs_sample, reward_sample, termination_sample = self.worldmodel.imagine_step(self.dreaming_batch_size, obs_sample, action_sample, reward_sample, termination_sample, imag_step)

                            pred_critic_obs = privileged_obs if privileged_obs is not None else pred_obs
                            pred_obs, pred_critic_obs = pred_obs.to(self.device,dtype=torch.float), pred_critic_obs.to(self.device,dtype=torch.float)
                            rewards, dones = reward_sample.to(self.device,dtype=torch.float).squeeze(), termination_sample.to(self.device,dtype=torch.float).squeeze()
                            # update the policy with the imagined data
                            self.alg.process_env_step_dream(rewards, dones, None)
                        
                        self.alg.compute_returns_dream(pred_critic_obs.squeeze())

                    # 4-2 update the policy
                    mean_value_loss, mean_surrogate_loss = self.alg.update_dream()
                    stop = time.time()
                    learn_time = stop - start


                # TODO: write log video in legged gym
                if it % self.log_movie_steps == 0:
                    log_video = True
                else:
                    log_video = False
                
                pass
            
            stop = time.time()
            learn_policy_w_WM_time = stop - start
            # =============end update policy on imagined data============= 

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

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
