import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, obs_shape, priv_obs_shape, action_shape, num_envs, max_length=128000, warmup_length=50000, store_on_gpu=False) -> None:
        self.store_on_gpu = store_on_gpu
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.float32, device="cuda", requires_grad=False)
            self.priv_obs_buffer = torch.empty((max_length//num_envs, num_envs, *priv_obs_shape), dtype=torch.float32, device="cuda", requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs, *action_shape), dtype=torch.float32, device="cuda", requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device="cuda", requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.bool, device="cuda", requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.float32)
            self.priv_obs_buffer = np.empty((max_length//num_envs, num_envs, *priv_obs_shape), dtype=np.float32)
            self.action_buffer = np.empty((max_length//num_envs, num_envs, *action_shape), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.bool8)        

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.full = False

    # def load_trajectory(self, path):
    #     buffer = pickle.load(open(path, "rb"))
    #     if self.store_on_gpu:
    #         self.external_buffer = {name: torch.from_numpy(buffer[name]).to("cuda") for name in buffer}
    #     else:
    #         self.external_buffer = buffer
    #     self.external_buffer_length = self.external_buffer["obs"].shape[0]

    # def sample_external(self, batch_size, batch_length, to_device="cuda"):
    #     indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
    #     if self.store_on_gpu:
    #         obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
    #         action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
    #         reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
    #         termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
    #     else:
    #         obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
    #         action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
    #         reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
    #         termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
    #     return obs, action, reward, termination

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, to_device="cuda"):
        if self.store_on_gpu:
            obs, priv_obs, action, reward, termination = [], [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(torch.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    priv_obs.append(torch.stack([self.priv_obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            # if self.external_buffer_length is not None and external_batch_size > 0:
            #     external_obs, external_action, external_reward, external_termination = self.sample_external(
            #         external_batch_size, batch_length, to_device)
            #     obs.append(external_obs)
            #     action.append(external_action)
            #     reward.append(external_reward)
            #     termination.append(external_termination)

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
        else:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(np.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(np.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(np.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(np.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            # if self.external_buffer_length is not None and external_batch_size > 0:
            #     external_obs, external_action, external_reward, external_termination = self.sample_external(
            #         external_batch_size, batch_length, to_device)
            #     obs.append(external_obs)
            #     action.append(external_action)
            #     reward.append(external_reward)
            #     termination.append(external_termination)

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float().cuda() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).cuda()
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).cuda()
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).cuda()

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.from_numpy(action)
            self.reward_buffer[self.last_pointer] = torch.from_numpy(reward)
            self.termination_buffer[self.last_pointer] = torch.from_numpy(termination)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs

# saves exeperience per time sequence
class ReplayBuffer_seq():
    def __init__(self, obs_shape, priv_obs_shape, action_shape, num_envs, num_steps_per_env, max_length=128000, warmup_length=50000, store_on_gpu=False, device = "cuda") -> None:
        self.store_on_gpu = store_on_gpu
        actual_max_length = (max_length//num_envs)*num_envs
        self.device = device if store_on_gpu else "cpu"
        self.obs_buffer = torch.empty((actual_max_length, num_steps_per_env, *obs_shape), dtype=torch.float32, device=self.device, requires_grad=False)
        self.priv_obs_buffer = torch.empty((actual_max_length, num_steps_per_env, *priv_obs_shape), dtype=torch.float32, device=self.device, requires_grad=False)
        self.action_buffer = torch.empty((actual_max_length, num_steps_per_env, *action_shape), dtype=torch.float32, device=self.device, requires_grad=False)
        self.reward_buffer = torch.empty((actual_max_length, num_steps_per_env), dtype=torch.float32, device=self.device, requires_grad=False)
        self.termination_buffer = torch.empty((actual_max_length, num_steps_per_env), dtype=torch.bool, device=self.device, requires_grad=False)
  
        self.actual_max_length = actual_max_length
        self.length = 0
        self.num_envs = num_envs
        self.current_index = 0
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.num_steps_per_env = num_steps_per_env
        self.full = False
        
    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, to_device="cuda"):
        max_index = self.actual_max_length if self.full else self.current_index
        indices = torch.randint(0, max_index, (batch_size,), device=to_device)

        obs = self.obs_buffer[indices, :batch_length]
        priv_obs = self.priv_obs_buffer[indices, :batch_length]
        action = self.action_buffer[indices, :batch_length]
        reward = self.reward_buffer[indices, :batch_length]
        termination = self.termination_buffer[indices, :batch_length]

        return obs, priv_obs, action, reward, termination

    def append(self, obs, priv_obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        # if the total length is reached, the first data is cleared and the new data is added
        # handle the overflow by wrapping around
        batch_size = obs.shape[0]
        end_index = self.current_index + batch_size

        if self.store_on_gpu:
            if end_index > self.actual_max_length:
                overflow = end_index - self.actual_max_length
                # store the buffer from current_index~end with obs from 0~batch_size-overflow
                self.obs_buffer[self.current_index:] = obs[:batch_size-overflow]
                self.priv_obs_buffer[self.current_index:] = priv_obs[:batch_size-overflow]
                self.action_buffer[self.current_index:] = action[:batch_size-overflow]
                self.reward_buffer[self.current_index:] = reward[:batch_size-overflow]
                self.termination_buffer[self.current_index:] = termination[:batch_size-overflow]
                # store the buffer from 0~overflow with obs from batch_size-overflow~end
                self.obs_buffer[:overflow] = obs[batch_size-overflow:]
                self.priv_obs_buffer[:overflow] = priv_obs[batch_size-overflow:]
                self.action_buffer[:overflow] = action[batch_size-overflow:]
                self.reward_buffer[:overflow] = reward[batch_size-overflow:]
                self.termination_buffer[:overflow] = termination[batch_size-overflow:]
            else:
                self.obs_buffer[self.current_index:end_index] = obs
                self.priv_obs_buffer[self.current_index:end_index] = priv_obs
                self.action_buffer[self.current_index:end_index] = action
                self.reward_buffer[self.current_index:end_index] = reward
                self.termination_buffer[self.current_index:end_index] = termination
        else:
            if end_index > self.actual_max_length:
                overflow = end_index - self.actual_max_length
                self.obs_buffer[self.current_index:] = obs[:batch_size-overflow]
                self.priv_obs_buffer[self.current_index:] = priv_obs[:batch_size-overflow]
                self.action_buffer[self.current_index:] = action[:batch_size-overflow]
                self.reward_buffer[self.current_index:] = reward[:batch_size-overflow]
                self.termination_buffer[self.current_index:] = termination[:batch_size-overflow]
                self.obs_buffer[:overflow] = obs[batch_size-overflow:]
                self.priv_obs_buffer[:overflow] = priv_obs[batch_size-overflow:]
                self.action_buffer[:overflow] = action[batch_size-overflow:]
                self.reward_buffer[:overflow] = reward[batch_size-overflow:]
                self.termination_buffer[:overflow] = termination[batch_size-overflow:]
            else:
                self.obs_buffer[self.current_index:end_index] = obs
                self.priv_obs_buffer[self.current_index:end_index] = priv_obs
                self.action_buffer[self.current_index:end_index] = action
                self.reward_buffer[self.current_index:end_index] = reward
                self.termination_buffer[self.current_index:end_index] = termination
        self.current_index += batch_size
        if end_index >= self.actual_max_length:
            self.current_index = end_index % self.actual_max_length
            self.full = True

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.actual_max_length if self.full else self.current_index

# saves exeperience per time sequence
class ReplayBuffer_seq_new():
    def __init__(self, obs_shape, priv_obs_shape, action_shape, num_envs, num_steps_per_env, device, max_length=128000, warmup_length=50000, store_on_gpu=False) -> None:
        self.store_on_gpu = store_on_gpu
        self.max_length = max_length
        self.device = device if store_on_gpu else "cpu"
        self.obs_buffer = torch.empty((max_length, num_steps_per_env, *obs_shape), dtype=torch.float32, device=self.device, requires_grad=False)
        self.priv_obs_buffer = torch.empty((max_length, num_steps_per_env, *priv_obs_shape), dtype=torch.float32, device=self.device, requires_grad=False)
        self.action_buffer = torch.empty((max_length, num_steps_per_env, *action_shape), dtype=torch.float32, device=self.device, requires_grad=False)
        self.reward_buffer = torch.empty((max_length, num_steps_per_env), dtype=torch.float32, device=self.device, requires_grad=False)
        self.termination_buffer = torch.empty((max_length, num_steps_per_env), dtype=torch.bool, device=self.device, requires_grad=False)
  
        self.length = 0
        self.num_envs = num_envs
        self.current_index = 0
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        self.num_steps_per_env = num_steps_per_env
        self.full = False
        
    def ready(self):
        return self.length > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, batch_length, to_device="cuda"):
        max_index = self.max_length if self.full else self.current_index
        indices = torch.randint(0, max_index, (batch_size,), device=to_device)
        time_indices = torch.randint(0, self.num_steps_per_env-batch_length, (batch_size,), device=to_device)
        obs = self.obs_buffer[indices[:, None], time_indices[:, None] + torch.arange(batch_length, device=to_device)]
        priv_obs = self.priv_obs_buffer[indices[:, None], time_indices[:, None] + torch.arange(batch_length, device=to_device)]
        action = self.action_buffer[indices[:, None], time_indices[:, None] + torch.arange(batch_length, device=to_device)]
        reward = self.reward_buffer[indices[:, None], time_indices[:, None] + torch.arange(batch_length, device=to_device)]
        termination = self.termination_buffer[indices[:, None], time_indices[:, None] + torch.arange(batch_length, device=to_device)]

        return obs, priv_obs, action, reward, termination

    def append(self, obs, priv_obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        # if the total length is reached, the first data is cleared and the new data is added
        # handle the overflow by wrapping around
        batch_size = obs.shape[0]
        end_index = self.current_index + batch_size

        if self.store_on_gpu:
            if end_index > self.max_length:
                overflow = end_index - self.max_length
                # store the buffer from current_index~end with obs from 0~batch_size-overflow
                self.obs_buffer[self.current_index:] = obs[:batch_size-overflow]
                self.priv_obs_buffer[self.current_index:] = priv_obs[:batch_size-overflow]
                self.action_buffer[self.current_index:] = action[:batch_size-overflow]
                self.reward_buffer[self.current_index:] = reward[:batch_size-overflow]
                self.termination_buffer[self.current_index:] = termination[:batch_size-overflow]
                # store the buffer from 0~overflow with obs from batch_size-overflow~end
                self.obs_buffer[:overflow] = obs[batch_size-overflow:]
                self.priv_obs_buffer[:overflow] = priv_obs[batch_size-overflow:]
                self.action_buffer[:overflow] = action[batch_size-overflow:]
                self.reward_buffer[:overflow] = reward[batch_size-overflow:]
                self.termination_buffer[:overflow] = termination[batch_size-overflow:]
            else:
                self.obs_buffer[self.current_index:end_index] = obs
                self.priv_obs_buffer[self.current_index:end_index] = priv_obs
                self.action_buffer[self.current_index:end_index] = action
                self.reward_buffer[self.current_index:end_index] = reward
                self.termination_buffer[self.current_index:end_index] = termination
        else:
            pass
        self.current_index += batch_size
        if end_index >= self.max_length:
            self.current_index = end_index % self.max_length
            self.full = True

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.max_length if self.full else self.current_index

