import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal, RelaxedOneHotCategorical
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import math
from .functions_losses import SymLogTwoHotLoss
from .attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from .transformer_model import StochasticTransformerKVCache, StochasticTransformerKVCache_small
from .agents import ActorCriticAgent


def compute_torques(actions, dof_pos_diff, dof_vel):
    actions_scaled = actions * 0.25
    # dof_pos_diff = -1*obs[9:21]/obs_scales.dof_pos
    # dof_vel = obs[21:33]/obs_scales.dof_vel
    dof_pos_diff = dof_pos_diff / 1.0
    dof_vel = dof_vel / 0.05
    torques = 20*(actions_scaled - dof_pos_diff) - 0.5*dof_vel
    return torques

class EncoderBN(nn.Module):
    def __init__(self, in_features, stem_features, latent_feature) -> None:
        super().__init__()
        # stem_channels = 64
        # final_feature_layers = 4
        # self.type = "conv"
        self.type = "linear"
        backbone = []
        # stem
        if self.type == "linear":
            backbone.append(
                nn.Linear(
                    in_features= in_features, 
                    out_features=stem_features,
                    bias=False
                )
        )
        if self.type == "conv":
            backbone.append(
                nn.Conv1d(
                    in_channels=in_features,
                    out_channels=stem_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
        feature_layers = math.ceil(math.log2(stem_features/latent_feature))# 1:64 #2:128 #3:256
        features = stem_features
        backbone.append(nn.BatchNorm1d(stem_features))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            if self.type == "linear":
                backbone.append(
                    nn.Linear(
                        in_features=features,
                        out_features=features//2,
                        # out_features=channels*2,
                        bias=False
                    )
                )
            if self.type == "conv":
                backbone.append(
                    nn.Conv1d(
                        in_channels=features,
                        out_channels=features*2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
            features /= 2
            features = int(features)
            # channels *=2
            feature_layers -= 1
            backbone.append(nn.BatchNorm1d(features))
            backbone.append(nn.ReLU(inplace=True))

            if feature_layers == 0:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_features = features

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B T F -> (B T) F")
        x = self.backbone(x)
        x = rearrange(x, "(B T) F -> B T F", B=batch_size)
        return x

class DecoderBN(nn.Module):
    def __init__(self, latent_dim, head_feature, output_features, stem_channels) -> None:
        super().__init__()
        self.type = "linear"
        # self.type = "conv"
        backbone = []
        # stem
        backbone.append(nn.Linear(latent_dim, head_feature, bias=False))
        backbone.append(Rearrange('B L F -> (B L) F', F=head_feature))
        backbone.append(nn.BatchNorm1d(head_feature))
        backbone.append(nn.ReLU(inplace=True))
        

        # layers
        features = head_feature
        while True:
            if features == stem_channels:
                break
            if self.type == "linear":
                backbone.append(
                    nn.Linear(
                        in_features=features,
                        out_features=features*2,
                        bias=False
                    )
                )
            if self.type == "conv":
                backbone.append(
                    nn.ConvTranspose1d(
                        in_channels=features,
                        out_channels=features//2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
            features *= 2
            backbone.append(nn.BatchNorm1d(features))
            backbone.append(nn.ReLU(inplace=True))
        # add final layer for boolean and float output
        bool_channels = 4
        self.float_head = nn.Linear(features, output_features-bool_channels)
        self.bool_head = nn.Linear(features, bool_channels)
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0] # B L (K C)
        sample = self.backbone(sample)
        float_out = self.float_head(sample)
        bool_logits = self.bool_head(sample)
        bool_out = torch.sigmoid(bool_logits)
        float_out = rearrange(float_out, "(B L) F -> B L F", B=batch_size)
        bool_out = rearrange(bool_out, "(B L) F -> B L F", B=batch_size)
        obs_hat = torch.cat([float_out, bool_out], dim=-1)
        return obs_hat


class DistHead_stoch(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, in_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(in_feat_dim, stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        # logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        # logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits
    
class DistHeadVAE(nn.Module):
    '''
    Dist: abbreviation of distribution for VAE
    '''
    def __init__(self, in_feat_dim, transformer_hidden_dim, hidden_dim) -> None:
        super().__init__()
        # TODO: add init_noise_std?
        self.stoch_dim = hidden_dim
        # post
        self.mu_in = nn.Linear(in_feat_dim, self.stoch_dim)
        self.log_var_in = nn.Linear(in_feat_dim, self.stoch_dim)
        # prior
        self.mu_out = nn.Linear(transformer_hidden_dim, self.stoch_dim)
        self.log_var_out = nn.Linear(transformer_hidden_dim, self.stoch_dim)
        # Initialize weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize log_var layers to produce values close to zero
        nn.init.constant_(self.log_var_in.bias, -5.0)  # Small initial variance
        nn.init.constant_(self.log_var_out.bias, -5.0)
        nn.init.xavier_uniform_(self.log_var_in.weight)
        nn.init.xavier_uniform_(self.log_var_out.weight)

        # Initialize mu layers
        nn.init.xavier_uniform_(self.mu_in.weight)
        nn.init.xavier_uniform_(self.mu_out.weight)
        if self.mu_in.bias is not None:
            nn.init.zeros_(self.mu_in.bias)
        if self.mu_out.bias is not None:
            nn.init.zeros_(self.mu_out.bias)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post_vae(self, x):
        mu_in = self.mu_in(x)
        log_var_in = self.log_var_in(x)
        return mu_in, log_var_in

    def forward_prior_vae(self, x):
        mu_out = self.mu_out(x)
        log_var_out = self.log_var_out(x)
        return mu_out, log_var_out


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        # squeeze reward
        reward = reward.squeeze(-1) # remove last 1 dim
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L F -> B L", "sum")
        return loss.mean()
    
class log_cosh_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = torch.log(torch.cosh(obs_hat - obs))
        loss = reduce(loss, "B L F -> B L", "sum")
        return loss.mean()
    
class MSELoss_GRU_hid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L F -> B", "sum")
        return loss.mean()
    

class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        # kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div

class KLDivLoss_normal(nn.Module):
    def __init__(self,free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits
    def forward(self, mu_post, std_p, mu_prior, std_q):
        std_p = torch.exp(0.5 * std_p)
        p_dist = Normal(mu_post, std_p)
        std_q = torch.exp(0.5 * std_q)
        q_dist = Normal(mu_prior, std_q)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div

class WorldModel(nn.Module):
    def __init__(self, in_channels, decoder_out_channels, action_dim, latent_feature,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, distribution,
                 lr=1e-4):
        super().__init__()
        
        self.dist_type = distribution
        self.transformer_hidden_dim = transformer_hidden_dim
        self.in_channels = in_channels
        self.decoder_out_channels = decoder_out_channels
        self.stoch_dim = 16
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.latent_feature = latent_feature

        self.encoder = EncoderBN(
            in_features=in_channels,
            stem_features=32,
            latent_feature=self.latent_feature
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1
        )
        if distribution == "vae":
            self.dist_head = DistHeadVAE(
                in_feat_dim=self.encoder.last_features,
                transformer_hidden_dim=transformer_hidden_dim,
                hidden_dim=self.stoch_dim
            )
        elif distribution == "stoch":
            self.dist_head = DistHead_stoch(
                in_feat_dim=self.encoder.last_features,
                transformer_hidden_dim=transformer_hidden_dim,
                stoch_dim=self.stoch_dim
            )
        self.image_decoder = DecoderBN(
            latent_dim = self.stoch_dim, 
            head_feature = self.stoch_dim, 
            output_features = self.decoder_out_channels, 
            stem_channels = 32
        )
        self.reward_decoder = RewardDecoder(
            num_classes=1,
            transformer_hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            transformer_hidden_dim=transformer_hidden_dim
        )

        self.mse_loss_func_obs = MSELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        # self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        if distribution == "vae":
            self.kl_div_loss = KLDivLoss_normal(free_bits=1)
        elif distribution == "stoch":
            self.kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)

    def encode_obs(self, obs,return_logits=False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            sample, post_logits = self.straight_throught_gradient(embedding,dist_type=self.dist_type,post=True, return_logits=True)
        if return_logits:
            return sample, post_logits
        return sample

    def calc_last_dist_feat(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            prior_sample = self.straight_throught_gradient(last_dist_feat,dist_type=self.dist_type, post=False)
        return prior_sample, last_dist_feat

    def predict_next(self, last_sample, return_logits=False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_sample)# transformer features

            # decoding
            prior_sample, prior_logits = self.straight_throught_gradient(dist_feat, dist_type=self.dist_type, post=False, return_logits=True)
            # get predictions
            obs_hat = self.image_decoder(prior_sample) # predicted next observation
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)
            # termination_hat = termination_hat > 0
        if return_logits:
            return obs_hat, reward_hat.unsqueeze(-1), termination_hat.unsqueeze(-1), prior_sample, dist_feat, prior_logits
        return obs_hat, reward_hat.unsqueeze(-1), termination_hat.unsqueeze(-1), prior_sample, dist_feat

    def straight_throught_gradient(self, logits, dist_type, post, return_logits=False):
        if dist_type == "vae":
            # normal distribution
            if post:
                # if posterier, use the post head
                mu, log_var = self.dist_head.forward_post_vae(logits)
            else:
                # if prior, use the prior head
                mu, log_var = self.dist_head.forward_prior_vae(logits)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + eps * std
            if return_logits:
                return sample, (mu, log_var)
        elif dist_type == "stoch":
            # categorical distribution
            if post:
                # if posterier, use the post head
                logits = self.dist_head.forward_post(logits)
            else:
                # if prior, use the prior head
                logits = self.dist_head.forward_prior(logits)
            dist = OneHotCategorical(logits=logits)
            # dist = RelaxedOneHotCategorical(logits=logits)
            sample = dist.sample() + dist.probs - dist.probs.detach()
            if return_logits:
                return sample, logits
        
        return sample

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            # these three parameters are used to predict the future, so it will have one extra
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            obs_hat_size = (imagine_batch_size, imagine_batch_length+1, self.decoder_out_channels) # set obs dimension
            # other parametes are saying information about the current time step
            action_size = (imagine_batch_size, imagine_batch_length, 12) # set action dimension
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.obs_hat_buffer = torch.zeros(obs_hat_size, dtype=dtype, device="cuda")
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def imagine_data(self, agent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, logger):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1]
            )
            last_action = sample_action[:, i:i+1]
        self.obs_hat_buffer[:, 0:1] = last_obs_hat
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat
        
        # imagine
        for i in range(imagine_batch_length):
            action = agent.act(torch.cat([self.obs_hat_buffer[:, i:i+1], last_action], dim=-1))
            self.action_buffer[:, i:i+1] = action
            last_action = action
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1])

            self.obs_hat_buffer[:, i+1:i+2] = last_obs_hat
            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat

            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat

        return self.obs_hat_buffer, self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def feed_context(self, batch_size, sample_obs, sample_action, batch_length,mode="train"):
        # if start to step using imagination, initialize the buffer and reset kv_cache
        # get the full observations or context observations
        # returns the first predicted observation, reward and termination, and the prior samples 
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        self.init_imagine_buffer(batch_size, batch_length, dtype=self.tensor_dtype)
        self.storm_transformer.reset_kv_cache_list(batch_size, dtype=self.tensor_dtype)
        # =====aggregate the kv_cache=====
        # context
        embedding = self.encoder(sample_obs)
        post_sample, post_logits = self.straight_throught_gradient(embedding,dist_type=self.dist_type,post=True, return_logits=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(post_sample)
            dist_feat = self.storm_transformer.forward_context(post_sample, sample_action, temporal_mask)
            prior_sample, prior_logits = self.straight_throught_gradient(dist_feat, dist_type=self.dist_type, post=False, return_logits=True)
            obs_hat = self.image_decoder(prior_sample)
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)
            # termination_hat = termination_hat > 0
        # TODO: remove the kv_cache_list after 32
        return obs_hat, reward_hat.unsqueeze(-1), termination_hat.unsqueeze(-1), prior_sample, prior_logits, post_sample, post_logits

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, return_loss=False):
        # update by one step
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(critic_obs[:,:,:45])
            post_sample, post_logits = self.straight_throught_gradient(embedding, dist_type=self.dist_type, post=True, return_logits=True)

            # decoding image
            obs_hat = self.image_decoder(post_sample)

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, post_sample.device)
            dist_feat = self.storm_transformer(post_sample, temporal_mask)

            prior_sample, prior_logits = self.straight_throught_gradient(dist_feat,dist_type=self.dist_type, return_logits=True, post=False)
            
            obs_hat_out = self.image_decoder(prior_sample)
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            # reconstruction loss for the observations, tokenizer update
            # float part
            reconstruction_loss_float = self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,:,:45])
            # boolean part
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,:,45:])
            reconstruction_loss = reconstruction_loss_float + reconstruction_loss_contact
            
            # reconstruction loss for the observations, predictor update, passes through the transformer
            # float part
            reconstruction_loss_out_float  = self.mse_loss_func_obs(obs_hat_out[:,:-1,:45], critic_obs[:,1:,:45])
            # boolean part
            reconstruction_loss_out_contact = self.bce_with_logits_loss_func(obs_hat_out[:,:-1,45:], critic_obs[:,1:,45:])
            reconstruction_loss_out = reconstruction_loss_out_float + reconstruction_loss_out_contact

            # reconstruction loss for the latent space, used for the dynamics and representation learning
            latent_reconstruction_loss = self.mse_loss_func_obs(prior_sample[:,:-1,:], post_sample[:,1:,:])

            reward_loss = self.mse_loss(reward_hat.unsqueeze(-1), reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat.unsqueeze(-1), termination.float())
            # dyn-rep loss
            if self.dist_type == "vae":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[0][:,1:,:].detach(), post_logits[1][:,1:,:].detach(), prior_logits[0][:,:-1,:], prior_logits[1][:,:-1,:])
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[0][:,1:,:], post_logits[1][:,1:,:], prior_logits[0][:,:-1,:].detach(), prior_logits[1][:,:-1,:].detach())
            elif self.dist_type == "stoch":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
           
            # total_loss
            total_loss = reconstruction_loss + reconstruction_loss_out + latent_reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss
        
        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # observation_difference = torch.abs(obs_hat - critic_obs)
        observation_difference =torch.abs(obs_hat - critic_obs).mean(dim=-1)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()
        torque_diff = observation_difference[33:45].mean()
        
        if writer is not None and it > 0:
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_out", reconstruction_loss_out.item(), it)
            writer.add_scalar("WorldModel/latent_reconstruction_loss", latent_reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reward_loss", reward_loss.item(), it)
            writer.add_scalar("WorldModel/termination_loss", termination_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_loss", dynamics_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/representation_loss", representation_loss.item(), it)
            writer.add_scalar("WorldModel/representation_real_kl_div", representation_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
            # observation difference
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_vel_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_gravity_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
            writer.add_scalar("obs_diff/torque_diff", torque_diff.item(), it)

        if return_loss:
            return reconstruction_loss
    
    def update_autoregressive(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, context_length=32, alpha = 1.0, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # pass through the full observations to get the first predicted observation, reward and termination, also get the ground truth prior samples
            obs_hat, reward_hat, termination_hat, prior_sample, prior_logits, post_sample_full, post_logits = self.feed_context(batch_size, obs[:,:,:45], action, batch_length=batch_length)
            # extract the first predicted observation, reward and termination
            obs_hats = obs_hat[:,context_length-1:context_length,:] 
            reward_hats = reward_hat[:,context_length-1:context_length,:] 
            termination_hats = termination_hat[:,context_length-1:context_length,:]
            prior_samples = prior_sample[:,context_length-1:context_length,:]
            if self.dist_type == "vae":
                mu_priors = prior_logits[0][:,context_length-1:context_length,:] 
                var_priors = prior_logits[1][:,context_length-1:context_length,:] 
            elif self.dist_type == "stoch":
                prior_logits = prior_logits[:,context_length-1:context_length,:]

            # initialize the losses
            total_loss = 0
            reconstruction_loss_out = 0
            latent_reconstruction_loss = 0
            reward_loss = 0
            termination_loss = 0
            dynamics_losses = 0
            real_dynamics_kl_dives = 0
            representation_losses = 0
            real_representation_kl_dives = 0

            # reconstruction loss for the first predicted observation, reward and termination
            reconstruction_loss_out += self.mse_loss_func_obs(obs_hats[:,-1:,:45], critic_obs[:,context_length:context_length+1,:45])
            reconstruction_loss_out += self.bce_with_logits_loss_func(obs_hats[:,-1:,45:], critic_obs[:,context_length:context_length+1,45:])
            
            # reconstruction loss for the reward and termination
            reward_loss += self.mse_loss(reward_hats[:,-1:,:], reward[:,context_length:context_length+1,:])
            termination_loss += self.bce_with_logits_loss_func(termination_hats[:,-1:,:], termination[:,context_length:context_length+1,:].float())
            
            # reconstruction loss for the latent space
            latent_reconstruction_loss += self.mse_loss_func_obs(prior_samples[:,-1:,:], post_sample_full[:,context_length:context_length+1,:])
            
            # dyn-rep loss
            if self.dist_type == "vae":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length:context_length+1,:].detach(), post_logits[1][:,context_length:context_length+1,:].detach(), mu_priors, var_priors)
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length:context_length+1,:], post_logits[1][:,context_length:context_length+1,:], mu_priors.detach(), var_priors.detach())
            elif self.dist_type == "stoch":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[:, context_length:context_length+1].detach(), prior_logits)
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[:, context_length:context_length+1], prior_logits.detach())
            # Store dyn-rep losses
            dynamics_losses += 0.5 * dynamics_loss
            real_dynamics_kl_dives += dynamics_real_kl_div
            representation_losses += 0.1 * representation_loss
            real_representation_kl_dives += representation_real_kl_div

            for i in range(batch_length-context_length-1):
                action_sample = action[:, context_length+i:context_length+i+1, :]
                torques = compute_torques(action_sample, obs_hats[:, -1:, 9:21], obs_hats[:, -1:, 21:33])
                pred_obs_full = torch.cat([ obs_hats[:, -1:, :33],       # First 9 elements of pred_obs
                                            action_sample], dim=-1)
                post_sample, _ = self.encode_obs(pred_obs_full,return_logits=True)
                obs_hat, reward_hat, termination_hat, prior_sample, dist_feat, prior_logit = self.predict_next(post_sample, return_logits=True)
                obs_hats = torch.cat([obs_hats, obs_hat], dim=1)
                reward_hats = torch.cat([reward_hats, reward_hat], dim=1)
                termination_hats = torch.cat([termination_hats, termination_hat], dim=1)
                prior_samples = torch.cat([prior_samples, prior_sample], dim=1)
                if self.dist_type == "vae":
                    mu_priors = torch.cat([mu_priors, prior_logit[0]], dim=1)
                    var_priors = torch.cat([var_priors, prior_logit[1]], dim=1)
                elif self.dist_type == "stoch":
                    prior_logits = torch.cat([prior_logits, prior_logit], dim=1)

                # reconstruction loss for the predicted observation, reward and termination
                reconstruction_loss_out += alpha**i * self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,context_length+1+i:context_length+2+i,:45])
                reconstruction_loss_out += alpha**i * self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,context_length+1+i:context_length+2+i,45:])
                
                # reconstruction loss for the reward and termination
                reward_loss += alpha**i * self.mse_loss(reward_hat, reward[:,context_length+1+i:context_length+2+i,:])
                termination_loss += alpha**i * self.bce_with_logits_loss_func(termination_hat, termination[:,context_length+1+i:context_length+2+i,:].float())
                
                # reconstruction loss for the latent space
                latent_reconstruction_loss += alpha**i * self.mse_loss_func_obs(prior_samples[:,-1:,:], post_sample_full[:,context_length+1+i:context_length+2+i,:])
                
                # dyn-rep loss
                if self.dist_type == "vae":
                    dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length+1+i:context_length+2+i,:].detach(), post_logits[1][:,context_length+1+i:context_length+2+i,:].detach()
                                                                           ,mu_priors[:,-1:,:], var_priors[:,-1:,:])
                    representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length+1+i:context_length+2+i,:], post_logits[1][:,context_length+1+i:context_length+2+i,:], 
                                                                                       mu_priors[:,-1:,:].detach(), mu_priors[:,-1:,:].detach())
                elif self.dist_type == "stoch":
                    dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[:, context_length+1+i:context_length+2+i, :].detach(), prior_logits[:, -1:, :])
                    representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[:, context_length+1+i:context_length+2+i,:], prior_logits[:, -1:,:].detach())
                # Store dyn-rep losses
                dynamics_losses += alpha**i * 0.5 * dynamics_loss
                real_dynamics_kl_dives += alpha**i *dynamics_real_kl_div
                representation_losses += alpha**i * 0.1 * representation_loss
                real_representation_kl_dives += alpha**i * representation_real_kl_div
                
            total_loss = reconstruction_loss_out + latent_reconstruction_loss + reward_loss + termination_loss + dynamics_losses + representation_losses
        
         # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # calculate the observation difference
        observation_difference =torch.abs(obs_hat - critic_obs).mean(dim=-1)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()
        torque_diff = observation_difference[33:45].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            writer.add_scalar("WorldModel/reconstruction_loss_out", reconstruction_loss_out.item(), it)
            writer.add_scalar("WorldModel/latent_reconstruction_loss", latent_reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reward_loss", reward_loss.item(), it)
            writer.add_scalar("WorldModel/termination_loss", termination_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_loss", dynamics_losses.item(), it)
            writer.add_scalar("WorldModel/dynamics_real_kl_div", real_dynamics_kl_dives.item(), it)
            writer.add_scalar("WorldModel/representation_loss", representation_losses.item(), it)
            writer.add_scalar("WorldModel/representation_real_kl_div", real_representation_kl_dives.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)

            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_vel_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_gravity_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
            writer.add_scalar("obs_diff/torque_diff", torque_diff.item(), it)

class WorldModel_no_rew_term(nn.Module):
    def __init__(self, in_channels, decoder_out_channels, action_dim, latent_feature,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, distribution,
                 lr=1e-4):
        super().__init__()
        
        self.dist_type = distribution
        self.transformer_hidden_dim = transformer_hidden_dim
        self.in_channels = in_channels
        self.decoder_out_channels = decoder_out_channels
        self.stoch_dim = 16
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.latent_feature = latent_feature

        self.encoder = EncoderBN(
            in_features=in_channels,
            stem_features=32,
            latent_feature=self.latent_feature
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1
        )
        if distribution == "vae":
            self.dist_head = DistHeadVAE(
                in_feat_dim=self.encoder.last_features,
                transformer_hidden_dim=transformer_hidden_dim,
                hidden_dim=self.stoch_dim
            )
        elif distribution == "stoch":
            self.dist_head = DistHead_stoch(
                in_feat_dim=self.encoder.last_features,
                transformer_hidden_dim=transformer_hidden_dim,
                stoch_dim=self.stoch_dim
            )
        self.image_decoder = DecoderBN(
            latent_dim = self.stoch_dim, 
            head_feature = self.stoch_dim, 
            output_features = self.decoder_out_channels, 
            stem_channels = 32
        )

        self.mse_loss_func_obs = MSELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        # self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        if distribution == "vae":
            self.kl_div_loss = KLDivLoss_normal(free_bits=1)
        elif distribution == "stoch":
            self.kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)

    def encode_obs(self, obs,return_logits=False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            sample, post_logits = self.straight_throught_gradient(embedding,dist_type=self.dist_type,post=True, return_logits=True)
        if return_logits:
            return sample, post_logits
        return sample

    def calc_last_dist_feat(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            prior_sample = self.straight_throught_gradient(last_dist_feat,dist_type=self.dist_type, post=False)
        return prior_sample, last_dist_feat

    def predict_next(self, last_sample, return_logits=False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_sample)# transformer features

            # decoding
            prior_sample, prior_logits = self.straight_throught_gradient(dist_feat, dist_type=self.dist_type, post=False, return_logits=True)
            # get predictions
            obs_hat = self.image_decoder(prior_sample) # predicted next observation
            # termination_hat = termination_hat > 0
        if return_logits:
            return obs_hat, prior_sample, dist_feat, prior_logits
        return obs_hat, prior_sample, dist_feat

    def straight_throught_gradient(self, logits, dist_type, post, return_logits=False):
        if dist_type == "vae":
            # normal distribution
            if post:
                # if posterier, use the post head
                mu, log_var = self.dist_head.forward_post_vae(logits)
            else:
                # if prior, use the prior head
                mu, log_var = self.dist_head.forward_prior_vae(logits)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + eps * std
            if return_logits:
                return sample, (mu, log_var)
        elif dist_type == "stoch":
            # categorical distribution
            if post:
                # if posterier, use the post head
                logits = self.dist_head.forward_post(logits)
            else:
                # if prior, use the prior head
                logits = self.dist_head.forward_prior(logits)
            dist = OneHotCategorical(logits=logits)
            # dist = RelaxedOneHotCategorical(logits=logits)
            sample = dist.sample() + dist.probs - dist.probs.detach()
            if return_logits:
                return sample, logits
        
        return sample

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            # these three parameters are used to predict the future, so it will have one extra
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            obs_hat_size = (imagine_batch_size, imagine_batch_length+1, self.decoder_out_channels) # set obs dimension
            # other parametes are saying information about the current time step
            action_size = (imagine_batch_size, imagine_batch_length, 12) # set action dimension
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.obs_hat_buffer = torch.zeros(obs_hat_size, dtype=dtype, device="cuda")
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def imagine_data(self, agent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, logger):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1]
            )
            last_action = sample_action[:, i:i+1]
        self.obs_hat_buffer[:, 0:1] = last_obs_hat
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat
        
        # imagine
        for i in range(imagine_batch_length):
            action = agent.act(torch.cat([self.obs_hat_buffer[:, i:i+1], last_action], dim=-1))
            self.action_buffer[:, i:i+1] = action
            last_action = action
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1])

            self.obs_hat_buffer[:, i+1:i+2] = last_obs_hat
            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat

            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat

        return self.obs_hat_buffer, self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def feed_context(self, batch_size, sample_obs, sample_action, batch_length,mode="train"):
        # if start to step using imagination, initialize the buffer and reset kv_cache
        # get the full observations or context observations
        # returns the first predicted observation, reward and termination, and the prior samples 
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        self.init_imagine_buffer(batch_size, batch_length, dtype=self.tensor_dtype)
        self.storm_transformer.reset_kv_cache_list(batch_size, dtype=self.tensor_dtype)
        # =====aggregate the kv_cache=====
        # context
        embedding = self.encoder(sample_obs)
        post_sample, post_logits = self.straight_throught_gradient(embedding,dist_type=self.dist_type,post=True, return_logits=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(post_sample)
            dist_feat = self.storm_transformer.forward_context(post_sample, sample_action, temporal_mask)
            prior_sample, prior_logits = self.straight_throught_gradient(dist_feat, dist_type=self.dist_type, post=False, return_logits=True)
            obs_hat = self.image_decoder(prior_sample)

        # TODO: remove the kv_cache_list after 32
        return obs_hat, prior_sample, prior_logits, post_sample, post_logits

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, return_loss=False):
        # update by one step
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(critic_obs[:,:,:45])
            post_sample, post_logits = self.straight_throught_gradient(embedding, dist_type=self.dist_type, post=True, return_logits=True)

            # decoding image
            obs_hat = self.image_decoder(post_sample)

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, post_sample.device)
            dist_feat = self.storm_transformer(post_sample, temporal_mask)

            prior_sample, prior_logits = self.straight_throught_gradient(dist_feat,dist_type=self.dist_type, return_logits=True, post=False)
            
            obs_hat_out = self.image_decoder(prior_sample)

            # env loss
            # reconstruction loss for the observations, tokenizer update
            # float part
            reconstruction_loss_float = self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,:,:45])
            # boolean part
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,:,45:])
            reconstruction_loss = reconstruction_loss_float + reconstruction_loss_contact
            
            # reconstruction loss for the observations, predictor update, passes through the transformer
            # float part
            reconstruction_loss_out_float  = self.mse_loss_func_obs(obs_hat_out[:,:-1,:45], critic_obs[:,1:,:45])
            # boolean part
            reconstruction_loss_out_contact = self.bce_with_logits_loss_func(obs_hat_out[:,:-1,45:], critic_obs[:,1:,45:])
            reconstruction_loss_out = reconstruction_loss_out_float + reconstruction_loss_out_contact

            # reconstruction loss for the latent space, used for the dynamics and representation learning
            latent_reconstruction_loss = self.mse_loss_func_obs(prior_sample[:,:-1,:], post_sample[:,1:,:])

            # dyn-rep loss
            if self.dist_type == "vae":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[0][:,1:,:].detach(), post_logits[1][:,1:,:].detach(), prior_logits[0][:,:-1,:], prior_logits[1][:,:-1,:])
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[0][:,1:,:], post_logits[1][:,1:,:], prior_logits[0][:,:-1,:].detach(), prior_logits[1][:,:-1,:].detach())
            elif self.dist_type == "stoch":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
           
            # total_loss
            total_loss = reconstruction_loss + reconstruction_loss_out + latent_reconstruction_loss + 0.5*dynamics_loss + 0.1*representation_loss
        
        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # observation_difference = torch.abs(obs_hat - critic_obs)
        observation_difference =torch.abs(obs_hat - critic_obs).mean(dim=-1)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()
        torque_diff = observation_difference[33:45].mean()
        
        if writer is not None and it > 0:
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_out", reconstruction_loss_out.item(), it)
            writer.add_scalar("WorldModel/latent_reconstruction_loss", latent_reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_loss", dynamics_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/representation_loss", representation_loss.item(), it)
            writer.add_scalar("WorldModel/representation_real_kl_div", representation_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
            # observation difference
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_vel_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_gravity_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
            writer.add_scalar("obs_diff/torque_diff", torque_diff.item(), it)

        if return_loss:
            return reconstruction_loss
    
    def update_autoregressive(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, context_length=32, alpha = 1.0, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # pass through the full observations to get the first predicted observation, reward and termination, also get the ground truth prior samples
            obs_hat, prior_sample, prior_logits, post_sample_full, post_logits = self.feed_context(batch_size, obs[:,:,:45], action, batch_length=batch_length)
            # extract the first predicted observation, reward and termination
            obs_hats = obs_hat[:,context_length-1:context_length,:] 
            prior_samples = prior_sample[:,context_length-1:context_length,:]
            if self.dist_type == "vae":
                mu_priors = prior_logits[0][:,context_length-1:context_length,:] 
                var_priors = prior_logits[1][:,context_length-1:context_length,:] 
            elif self.dist_type == "stoch":
                prior_logits = prior_logits[:,context_length-1:context_length,:]

            # initialize the losses
            total_loss = 0
            reconstruction_loss_out = 0
            latent_reconstruction_loss = 0
            dynamics_losses = 0
            real_dynamics_kl_dives = 0
            representation_losses = 0
            real_representation_kl_dives = 0

            # reconstruction loss for the first predicted observation
            reconstruction_loss_out += self.mse_loss_func_obs(obs_hats[:,-1:,:45], critic_obs[:,context_length:context_length+1,:45])
            reconstruction_loss_out += self.bce_with_logits_loss_func(obs_hats[:,-1:,45:], critic_obs[:,context_length:context_length+1,45:])
            
            # reconstruction loss for the latent space
            latent_reconstruction_loss += self.mse_loss_func_obs(prior_samples[:,-1:,:], post_sample_full[:,context_length:context_length+1,:])
            
            # dyn-rep loss
            if self.dist_type == "vae":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length:context_length+1,:].detach(), post_logits[1][:,context_length:context_length+1,:].detach(), mu_priors, var_priors)
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length:context_length+1,:], post_logits[1][:,context_length:context_length+1,:], mu_priors.detach(), var_priors.detach())
            elif self.dist_type == "stoch":
                dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[:, context_length:context_length+1].detach(), prior_logits)
                representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[:, context_length:context_length+1], prior_logits.detach())
            # Store dyn-rep losses
            dynamics_losses += 0.5 * dynamics_loss
            real_dynamics_kl_dives += dynamics_real_kl_div
            representation_losses += 0.1 * representation_loss
            real_representation_kl_dives += representation_real_kl_div

            for i in range(batch_length-context_length-1):
                action_sample = action[:, context_length+i:context_length+i+1, :]
                torques = compute_torques(action_sample, obs_hats[:, -1:, 9:21], obs_hats[:, -1:, 21:33])
                pred_obs_full = torch.cat([ obs_hats[:, -1:, :33],       # First 9 elements of pred_obs
                                            action_sample], dim=-1)
                post_sample, _ = self.encode_obs(pred_obs_full,return_logits=True)
                obs_hat, prior_sample, dist_feat, prior_logit = self.predict_next(post_sample, return_logits=True)
                obs_hats = torch.cat([obs_hats, obs_hat], dim=1)
                prior_samples = torch.cat([prior_samples, prior_sample], dim=1)
                if self.dist_type == "vae":
                    mu_priors = torch.cat([mu_priors, prior_logit[0]], dim=1)
                    var_priors = torch.cat([var_priors, prior_logit[1]], dim=1)
                elif self.dist_type == "stoch":
                    prior_logits = torch.cat([prior_logits, prior_logit], dim=1)

                # reconstruction loss for the predicted observation
                reconstruction_loss_out += alpha**i * self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,context_length+1+i:context_length+2+i,:45])
                reconstruction_loss_out += alpha**i * self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,context_length+1+i:context_length+2+i,45:])
                
                # reconstruction loss for the latent space
                latent_reconstruction_loss += alpha**i * self.mse_loss_func_obs(prior_samples[:,-1:,:], post_sample_full[:,context_length+1+i:context_length+2+i,:])
                
                # dyn-rep loss
                if self.dist_type == "vae":
                    dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length+1+i:context_length+2+i,:].detach(), post_logits[1][:,context_length+1+i:context_length+2+i,:].detach()
                                                                           ,mu_priors[:,-1:,:], var_priors[:,-1:,:])
                    representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[0][:,context_length+1+i:context_length+2+i,:], post_logits[1][:,context_length+1+i:context_length+2+i,:], 
                                                                                       mu_priors[:,-1:,:].detach(), mu_priors[:,-1:,:].detach())
                elif self.dist_type == "stoch":
                    dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(post_logits[:, context_length+1+i:context_length+2+i, :].detach(), prior_logits[:, -1:, :])
                    representation_loss, representation_real_kl_div = self.kl_div_loss(post_logits[:, context_length+1+i:context_length+2+i,:], prior_logits[:, -1:,:].detach())
                # Store dyn-rep losses
                dynamics_losses += alpha**i * 0.5 * dynamics_loss
                real_dynamics_kl_dives += alpha**i *dynamics_real_kl_div
                representation_losses += alpha**i * 0.1 * representation_loss
                real_representation_kl_dives += alpha**i * representation_real_kl_div
                
            total_loss = reconstruction_loss_out + latent_reconstruction_loss + dynamics_losses + representation_losses
        
         # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # calculate the observation difference
        observation_difference =torch.abs(obs_hat - critic_obs).mean(dim=-1)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()
        torque_diff = observation_difference[33:45].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            writer.add_scalar("WorldModel/reconstruction_loss_out", reconstruction_loss_out.item(), it)
            writer.add_scalar("WorldModel/latent_reconstruction_loss", latent_reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_loss", dynamics_losses.item(), it)
            writer.add_scalar("WorldModel/dynamics_real_kl_div", real_dynamics_kl_dives.item(), it)
            writer.add_scalar("WorldModel/representation_loss", representation_losses.item(), it)
            writer.add_scalar("WorldModel/representation_real_kl_div", real_representation_kl_dives.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)

            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_vel_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_gravity_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
            writer.add_scalar("obs_diff/torque_diff", torque_diff.item(), it)

class WorldModel_GRU(nn.Module):
    def __init__(self, obs_dim, decoder_out_channels, gru_hidden_size=256, mlp_hidden_size=128):
        """
        Args:
            in_channels (int): Number of input features (observations).
            gru_hidden_size (int): Hidden size of the GRU.
            mlp_hidden_size (int): Hidden size of the MLP layers.
            decoder_out_channels (int): Number of output features (predicted observations).
        """
        super().__init__()
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.obs_dim = obs_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.decoder_out_channels = decoder_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.bool_channles = 12
        # GRU: Processes sequential data
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=gru_hidden_size, num_layers=2, batch_first=True)
        
        # MLP heads for observation and contact prediction (mean and variance)
        self.obs_head = nn.Sequential(
            nn.Linear(gru_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        self.bool_head = nn.Linear(self.mlp_hidden_size, self.bool_channles)
        self.float_head = nn.Linear(self.mlp_hidden_size, 2*(decoder_out_channels - self.bool_channles))
        self.mse_loss_func_obs = MSELoss()
        self.mse_loss_hidden = MSELoss_GRU_hid()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=800, gamma=0.1)

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var) + 1e-6
        # Generate random noise using the same shape as std
        q = Normal(mu, std)
        # Return the reparameterized sample
        return q.rsample()
    
    def forward(self, obs_sequence, h = None):
        """
        Forward pass for the GRU-based world model.

        Args:
            obs (Tensor): Input observations of shape (batch_size, seq_length, in_channels).
            action (Tensor): Actions of shape (batch_size, seq_length, action_dim).
            hidden_state (Tensor, optional): Initial hidden state for the GRU.

        Returns:
            obs_hat (Tensor): Predicted next observations of shape (batch_size, seq_length, decoder_out_channels).
            reward_hat (Tensor): Predicted rewards of shape (batch_size, seq_length, 1).
            termination_hat (Tensor): Predicted termination signals of shape (batch_size, seq_length, 1).
            hidden_state (Tensor): Final hidden state of the GRU.
        """
        # Encode input features
        if h is None:
            out, h = self.gru(obs_sequence)
        else:
            out, h = self.gru(obs_sequence, h)
        obs_hat = self.obs_head(out) # obs_hat: (batch, 1, input_dim)
        obs_dist = self.float_head(obs_hat)
        bool_logits = self.bool_head(obs_hat)
        bool_out = torch.sigmoid(bool_logits)
        mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
        obs_hat = self.reparameterize(mu_o, logvar_o)
        obs_hat = torch.cat([obs_hat, bool_out], dim=-1)

        return obs_hat, h

    def autoregressive_training_step(self, obs, critic_obs, actions, alpha=1.0, context_length=32, pred_length=8,writer=None,it=0):
        """
        model: RWMGRUWorldModel
        obs: (B, T, D)
        act: (B, T, A)
        contact: (B, T, 8)
        """
        B, T, _ = obs.shape
        total_loss = 0
        count = 0
        obs_hats=[]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # obs_context = obs[:, :32, :] # (B, 32, D)
            h_full = []
            # get the hidden state for the 40 steps
            obs_hat, h = self.forward(critic_obs[:, 0:1, :45]) # (B, 32, D)
            h_full.append(h)
            for i in range(context_length+pred_length-1):
                obs_hat, h = self.forward(critic_obs[:, i+1:i+2, :45], h) # (B, 1, D)
                h_full.append(h)
            # the full hidden states for the 40 steps
            h_full = torch.stack(h_full) # (B, 32, D)
            # get the first input for the autoregressive model
            input_t = critic_obs[:,31:32,:45]
            h = h_full[31] # (B, 1, D)
            for i in range(pred_length):
                # predict the 32+i step
                obs_hat,h = self.forward(input_t, h) # out: (batch, 1, hidden_dim)
                action_sample = actions[:, 32+i:32+1+i, :] # (batch, 1, action_dim)
                torques = compute_torques(action_sample, obs_hat[:, :, 9:21], obs_hat[:, :, 21:33])
                input_t = torch.cat([obs_hat[:, :, :33],       # First 9 elements of pred_obs
                                     torques
                                    ], dim=-1)
                obs_hats.append(obs_hat)
                # loss for continuous states
                total_loss += alpha**i * self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,32+i:32+i+1,:45])
                # loss for contact states
                total_loss += alpha**i * self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,32+i:32+i+1,45:])
                # loss for latent states
                total_loss += alpha**i * self.mse_loss_hidden(h.permute(1,0,2),h_full[32+i].permute(1,0,2))
                # use prediction as next input
         # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        obs_hats = torch.cat(obs_hats, dim=1)
        observation_difference = torch.abs(obs_hats - critic_obs[:,-8:,:])
        observation_difference = torch.flatten(observation_difference,0,1).mean(dim=0)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            # writer.add_scalar("autoregresser/reconstruction_loss", reconstruction_loss.item(), it)
            # writer.add_scalar("autoregresser/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("gru_autoregresser/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)

class WorldModel_normal_small(nn.Module):
    def __init__(self, in_channels, decoder_out_channels,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads,
                 lr = 1e-3, observation_head = False):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.decoder_out_channels = decoder_out_channels
        self.mlp_hidden_size = 128
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.in_channels = in_channels
        self.bool_channles = 12
        if observation_head:
            self.encoder = nn.Linear(in_channels, transformer_hidden_dim)
            in_channels = transformer_hidden_dim
        self.storm_transformer = StochasticTransformerKVCache_small(
            input_dim=in_channels,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.0
        )
        self.obs_head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, self.mlp_hidden_size),
            nn.ReLU()
        )
        self.bool_head = nn.Linear(self.mlp_hidden_size, self.bool_channles)
        self.float_head = nn.Linear(self.mlp_hidden_size, 2*(decoder_out_channels - self.bool_channles))
        self.mse_loss = MSELoss()
        self.logcosh_loss = log_cosh_loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.kl_div_loss = KLDivLoss_normal(free_bits = 1.0)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr,weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(last_dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, sample):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            trans_feat = self.storm_transformer.forward_with_kv_cache(sample)# transformer features
            obs_hat = self.obs_head(trans_feat)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat
    
    def predict_next_without_kv_cache(self, sample, mask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            trans_feat = self.storm_transformer.forward(sample, mask) # transformer features
            obs_hat = self.obs_head(trans_feat)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat

    def predict_next_without_kv_cache_with_attention(self, sample, mask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            trans_feat, attns = self.storm_transformer.forward_with_attention(sample, mask) # transformer features
            obs_hat = self.obs_head(trans_feat)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat, attns

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var) + 1e-6
        # Generate random noise using the same shape as std
        q = Normal(mu, std)
        # Return the reparameterized sample
        return q.rsample()
    
    def compute_torques(self, actions, dof_pos_diff, dof_vel):
        actions_scaled = actions * 0.25
        # dof_pos_diff = -1*obs[9:21]/obs_scales.dof_pos
        # dof_vel = obs[21:33]/obs_scales.dof_vel
        dof_pos_diff = dof_pos_diff / 1.0
        dof_vel = dof_vel / 0.05
        torques = 20*(actions_scaled - dof_pos_diff) - 0.5*dof_vel
        return torques

    def setup_imagination_train(self, batch_size, sample_obs, batch_length):
        # if start to step using imagination, initialize the buffer and reset kv_cache
        self.storm_transformer.reset_kv_cache_list(batch_size, dtype=self.tensor_dtype)
        self.storm_transformer.eval()
        batch_size, batch_length = sample_obs.shape[:2]
        # =====aggregate the kv_cache=====
        # context
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, device=sample_obs.device)
            trans_feat = self.storm_transformer.forward_context(sample_obs, temporal_mask)
            obs_hat = self.obs_head(trans_feat) # obs_hat: (batch, 1, input_dim)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)

        return obs_hat[:, -1:,:], trans_feat[:, -1:,:]

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # modify the obs to not include command and actions
        # which is obs[:, :, 9:12] and obs[:, :, 36:48]
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36], obs[:, :, 48:]], dim=-1)
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36]], dim=-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, obs.device)
            # get transformer features
            obs_hat, trans_feat = self.predict_next_without_kv_cache(critic_obs[:,:,:45],temporal_mask)

            # env loss
            # reconstruction_loss for continuous states
            reconstruction_loss = self.mse_loss(obs_hat[:,:-1,:45], critic_obs[:,1:,:45])
            # reconstruction_loss for contact states
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:-1,45:], critic_obs[:,1:,45:])

            reconstruction_loss = reconstruction_loss + reconstruction_loss_contact
            total_loss = reconstruction_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        observation_difference = torch.abs(obs_hat - critic_obs)
        observation_difference = torch.flatten(observation_difference,0,1).mean(dim=0)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            # writer.add_scalar("WorldModel/real_kl_loss", real_kl_loss.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
        if return_loss:
            return reconstruction_loss
    
    def update_autoregressive(self, obs, critic_obs, action, context_length, pred_length, it, writer: SummaryWriter, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        context_length = 32
        prediction_horizon = 8
        total_loss = 0
        obs_hats = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, obs.device)
            # get transformer features
            obs_hat, trans_feat_full = self.predict_next_without_kv_cache(critic_obs[:,:,:45],temporal_mask)

            alpha = 1.0
            # get the context input for the autoregressive model
            obs_inputs = critic_obs[:,:32,:45]
            for i in range(prediction_horizon):
                # action_sample = action[:, 32+i:32+i+1, :]
                # pred_obs_full = torch.cat([obs_hat[:, :, :9],       # First 9 elements of pred_obs
                #                         obs[:,32+i:32+i+1,9:12],               # cmd_tensor
                #                             obs_hat[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
                #                             action_sample
                #                             ], dim=-1)
                temporal_mask = get_subsequent_mask_with_batch_length(context_length+i, obs.device)
                obs_hat, trans_feat = self.predict_next_without_kv_cache(obs_inputs,temporal_mask)
                torques = self.compute_torques(action[:, 32+i:32+i+1, :], obs_hat[:, -1:, 9:21], obs_hat[:, -1:, 21:33])
                obs_hat_for_input = torch.cat([obs_hat[:, -1:, :33],       # First 9 elements of pred_obs
                                     torques,   # Remaining elements of pred_obs (from index 9 to 32)
                                    ], dim=-1)
                obs_inputs = torch.cat([obs_inputs, obs_hat_for_input], dim=1)
                obs_hats.append(obs_hat[:,-1:,:])
                # loss for continuous states
                total_loss += alpha**i * self.mse_loss(obs_hat[:,-1:,:45], critic_obs[:,32+i:32+i+1,:45])
                # loss for contact states
                total_loss += alpha**i * self.bce_with_logits_loss_func(obs_hat[:,-1:,45:], critic_obs[:,32+i:32+i+1,45:])
                # loss for latent states
                total_loss += alpha**i * self.mse_loss(trans_feat[:,-1:,:],trans_feat_full[:,32+i:32+i+1,:])
                # total_loss += alpha**i * self.mse_loss(trans_feat, trans_feat_full[:,33+i:33+i+1,:].detach())

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        obs_hats = torch.cat(obs_hats, dim=1)
        observation_difference = torch.abs(obs_hats - critic_obs[:,-8:,:])
        observation_difference = torch.flatten(observation_difference,0,1).mean(dim=0)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            # writer.add_scalar("autoregresser/reconstruction_loss", reconstruction_loss.item(), it)
            # writer.add_scalar("autoregresser/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("autoregresser/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
        if return_loss:
            return total_loss.detach().cpu()

class WorldModel_normal_small_test(nn.Module):
    def __init__(self, in_channels, decoder_out_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.decoder_out_channels = decoder_out_channels
        self.mlp_hidden_size = 128
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.in_channels = in_channels
        self.bool_channles = 12
        self.hidden_dim = 64
        self.dist_head_vae = DistHeadVAE(
            image_feat_dim=in_channels,
            transformer_hidden_dim=transformer_hidden_dim,
            hidden_dim=self.hidden_dim
        )
        self.storm_transformer = StochasticTransformerKVCache_small(
            input_dim=self.hidden_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.0
        )
        self.obs_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.mlp_hidden_size),
            nn.ReLU()
        )
        self.bool_head = nn.Linear(self.mlp_hidden_size, self.bool_channles)
        self.float_head = nn.Linear(self.mlp_hidden_size, 2*(decoder_out_channels - self.bool_channles))
        self.mse_loss = MSELoss()
        self.logcosh_loss = log_cosh_loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.kl_div_loss = KLDivLoss_normal(free_bits = 1.0)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3,weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(last_dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
        return prior_flattened_sample, last_dist_feat
    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var) + 1e-6
        # Generate random noise using the same shape as std
        q = Normal(mu, std)
        # Return the reparameterized sample
        return q.rsample()
    
    def predict_next(self, sample):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            trans_feat = self.storm_transformer.forward_with_kv_cache(sample)# transformer features
            obs_hat = self.obs_head(trans_feat)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat
    
    def predict_next_without_kv_cache(self, sample, mask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            trans_feat = self.storm_transformer.forward(sample, mask) # transformer features
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(trans_feat)
            prior_sample = self.reparameterize(mu_prior, var_prior)
            obs_hat = self.obs_head(prior_sample) # obs_hat: (batch, 1, input_dim)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat, prior_sample

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var) + 1e-6
        # Generate random noise using the same shape as std
        q = Normal(mu, std)
        # Return the reparameterized sample
        return q.rsample()
    
    def compute_torques(self, actions, dof_pos_diff, dof_vel):
        actions_scaled = actions * 0.25
        # dof_pos_diff = -1*obs[9:21]/obs_scales.dof_pos
        # dof_vel = obs[21:33]/obs_scales.dof_vel
        dof_pos_diff = dof_pos_diff / 1.0
        dof_vel = dof_vel / 0.05
        torques = 20*(actions_scaled - dof_pos_diff) - 0.5*dof_vel
        return torques

    def setup_imagination_train(self, batch_size, sample_obs, batch_length):
        # if start to step using imagination, initialize the buffer and reset kv_cache
        self.storm_transformer.reset_kv_cache_list(batch_size, dtype=self.tensor_dtype)
        self.storm_transformer.eval()
        batch_size, batch_length = sample_obs.shape[:2]
        # =====aggregate the kv_cache=====
        # context
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, device=sample_obs.device)
            trans_feat = self.storm_transformer.forward_context(sample_obs, temporal_mask)
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(trans_feat)
            prior_sample = self.reparameterize(mu_prior, var_prior)
            obs_hat = self.obs_head(prior_sample) # obs_hat: (batch, 1, input_dim)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)

        return obs_hat[:, -1:,:], trans_feat[:, -1:,:]

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # modify the obs to not include command and actions
        # which is obs[:, :, 9:12] and obs[:, :, 36:48]
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36], obs[:, :, 48:]], dim=-1)
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36]], dim=-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            mu_post, var_post = self.dist_head_vae.forward_post_vae(critic_obs[:,:,:45])
            sample = self.reparameterize(mu_post, var_post)
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, obs.device)
            # get transformer features
            obs_hat, trans_feat, prior_sample = self.predict_next_without_kv_cache(sample,temporal_mask)

            # env loss
            # reconstruction_loss for continuous states
            reconstruction_loss = self.mse_loss(obs_hat[:,:-1,:45], critic_obs[:,1:,:45])
            # reconstruction_loss for contact states
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:-1,45:], critic_obs[:,1:,45:])
            # reconstruction_loss for latent states
            reconstruction_loss_latent = self.mse_loss(prior_sample[:,:-1,: ], sample[:,1:,:])
            reconstruction_loss = reconstruction_loss + reconstruction_loss_contact + reconstruction_loss_latent
            
            # dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(mu_post[:,1:,:].detach(), var_post[:,1:,:].detach(), mu_prior[:,:-1,:], var_prior[:,:-1,:])
            # representation_loss, representation_real_kl_div = self.kl_div_loss(mu_post[:,1:,:], var_post[:,1:,:], mu_prior[:,:-1,:].detach(), var_prior[:,:-1,:].detach())

            total_loss = reconstruction_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        observation_difference = torch.abs(obs_hat - critic_obs)
        observation_difference = torch.flatten(observation_difference,0,1).mean(dim=0)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            # writer.add_scalar("WorldModel/real_kl_loss", real_kl_loss.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
        if return_loss:
            return reconstruction_loss
    
    def update_autoregressive(self, obs, critic_obs, action, context_length, pred_length, it, writer: SummaryWriter):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        context_length = 32
        prediction_horizon = 8
        total_loss = 0
        obs_hats = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            mu_post, var_post = self.dist_head_vae.forward_post_vae(critic_obs[:,:,:45])
            samples = self.reparameterize(mu_post, var_post)
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, obs.device)
            # get transformer features
            obs_hat, trans_feat_full, prior_samples_full = self.predict_next_without_kv_cache(samples,temporal_mask)

            alpha = 1.0
            # get the context input for the autoregressive model
            obs_inputs = critic_obs[:,:32,:45]
            for i in range(prediction_horizon):
                mu_post, var_post = self.dist_head_vae.forward_post_vae(obs_inputs)
                samples = self.reparameterize(mu_post, var_post)
                temporal_mask = get_subsequent_mask_with_batch_length(context_length+i, obs.device)
                obs_hat, trans_feat, prior_sample = self.predict_next_without_kv_cache(samples,temporal_mask)
                torques = self.compute_torques(action[:, 32+i:32+i+1, :], obs_hat[:, -1:, 9:21], obs_hat[:, -1:, 21:33])
                obs_hat_for_input = torch.cat([obs_hat[:, -1:, :33],       # First 9 elements of pred_obs
                                     torques,   # Remaining elements of pred_obs (from index 9 to 32)
                                    ], dim=-1)
                obs_inputs = torch.cat([obs_inputs, obs_hat_for_input], dim=1)
                obs_hats.append(obs_hat[:,-1:,:])
                # loss for continuous states
                total_loss += alpha**i * self.mse_loss(obs_hat[:,-1:,:45], critic_obs[:,32+i:32+i+1,:45])
                # loss for contact states
                total_loss += alpha**i * self.bce_with_logits_loss_func(obs_hat[:,-1:,45:], critic_obs[:,32+i:32+i+1,45:])
                # loss for latent states
                total_loss += alpha**i * self.mse_loss(prior_sample[:,-1:,:],prior_samples_full[:,32+i:32+i+1,:])
                # total_loss += alpha**i * self.mse_loss(trans_feat, trans_feat_full[:,33+i:33+i+1,:].detach())
                # loss for kl divergence
                # total_loss += alpha**i * self.kl_div_loss(mu_post[:,1:,:], var_post[:,1:,:], mu_prior[:,:-1,:].detach(), var_prior[:,:-1,:].detach())[0]



        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        obs_hats = torch.cat(obs_hats, dim=1)
        observation_difference = torch.abs(obs_hats - critic_obs[:,-8:,:])
        observation_difference = torch.flatten(observation_difference,0,1).mean(dim=0)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            # writer.add_scalar("autoregresser/reconstruction_loss", reconstruction_loss.item(), it)
            # writer.add_scalar("autoregresser/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("autoregresser/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)

class WorldModel_normal_small_torch(nn.Module):
    def __init__(self, in_channels, decoder_out_channels,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads,
                 lr = 1e-3, observation_head = False):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.decoder_out_channels = decoder_out_channels
        self.mlp_hidden_size = 128
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.in_channels = in_channels
        self.bool_channles = 12
        
        # self.storm_transformer = StochasticTransformerKVCache_small(
        #     input_dim=in_channels,
        #     feat_dim=transformer_hidden_dim,
        #     num_layers=transformer_num_layers,
        #     num_heads=transformer_num_heads,
        #     max_length=transformer_max_length,
        #     dropout=0.0
        # )
        self.obs_in_head = nn.Linear(in_channels, transformer_hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_hidden_dim,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.storm_transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=transformer_num_layers)   
        self.obs_head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, self.mlp_hidden_size),
            nn.ReLU()
        )
        self.bool_head = nn.Linear(self.mlp_hidden_size, self.bool_channles)
        self.float_head = nn.Linear(self.mlp_hidden_size, 2*(decoder_out_channels - self.bool_channles))
        self.mse_loss = MSELoss()
        self.logcosh_loss = log_cosh_loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.kl_div_loss = KLDivLoss_normal(free_bits = 1.0)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr,weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(last_dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, sample):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            trans_feat = self.storm_transformer.forward_with_kv_cache(sample)# transformer features
            obs_hat = self.obs_head(trans_feat)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat
    
    def predict_next_without_kv_cache(self, sample, mask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            sample = self.obs_in_head(sample)  # project input to transformer hidden dim
            # trans_feat = self.storm_transformer.forward(sample, mask) # transformer features
            trans_feat = self.storm_transformer(
                tgt=sample, 
                memory=sample, 
                tgt_mask=mask, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None
            )
            obs_hat = self.obs_head(trans_feat)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat

    def predict_next_without_kv_cache_with_attention(self, sample, mask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            trans_feat, attns = self.storm_transformer.forward_with_attention(sample, mask) # transformer features
            obs_hat = self.obs_head(trans_feat)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)
        return obs_hat, trans_feat, attns

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var) + 1e-6
        # Generate random noise using the same shape as std
        q = Normal(mu, std)
        # Return the reparameterized sample
        return q.rsample()
    
    def compute_torques(self, actions, dof_pos_diff, dof_vel):
        actions_scaled = actions * 0.25
        # dof_pos_diff = -1*obs[9:21]/obs_scales.dof_pos
        # dof_vel = obs[21:33]/obs_scales.dof_vel
        dof_pos_diff = dof_pos_diff / 1.0
        dof_vel = dof_vel / 0.05
        torques = 20*(actions_scaled - dof_pos_diff) - 0.5*dof_vel
        return torques

    def setup_imagination_train(self, batch_size, sample_obs, batch_length):
        # if start to step using imagination, initialize the buffer and reset kv_cache
        self.storm_transformer.reset_kv_cache_list(batch_size, dtype=self.tensor_dtype)
        self.storm_transformer.eval()
        batch_size, batch_length = sample_obs.shape[:2]
        # =====aggregate the kv_cache=====
        # context
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, device=sample_obs.device)
            trans_feat = self.storm_transformer.forward_context(sample_obs, temporal_mask)
            obs_hat = self.obs_head(trans_feat) # obs_hat: (batch, 1, input_dim)
            obs_dist = self.float_head(obs_hat)
            bool_logits = self.bool_head(obs_hat)
            bool_out = torch.sigmoid(bool_logits)
            mu_o, logvar_o = torch.chunk(obs_dist, 2, dim=-1)
            obs_hat = self.reparameterize(mu_o, logvar_o)
            obs_hat = torch.cat([obs_hat, bool_out], dim=-1)

        return obs_hat[:, -1:,:], trans_feat[:, -1:,:]

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # modify the obs to not include command and actions
        # which is obs[:, :, 9:12] and obs[:, :, 36:48]
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36], obs[:, :, 48:]], dim=-1)
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36]], dim=-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, obs.device)
            # get transformer features
            obs_hat, trans_feat = self.predict_next_without_kv_cache(critic_obs[:,:,:45],temporal_mask)

            # env loss
            # reconstruction_loss for continuous states
            reconstruction_loss = self.mse_loss(obs_hat[:,:-1,:45], critic_obs[:,1:,:45])
            # reconstruction_loss for contact states
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:-1,45:], critic_obs[:,1:,45:])

            reconstruction_loss = reconstruction_loss + reconstruction_loss_contact
            total_loss = reconstruction_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        observation_difference = torch.abs(obs_hat - critic_obs)
        observation_difference = torch.flatten(observation_difference,0,1).mean(dim=0)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            # writer.add_scalar("WorldModel/real_kl_loss", real_kl_loss.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
        if return_loss:
            return reconstruction_loss
    
    def update_autoregressive(self, obs, critic_obs, action, context_length, pred_length, it, writer: SummaryWriter, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        context_length = 32
        prediction_horizon = 8
        total_loss = 0
        obs_hats = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, obs.device)
            # get transformer features
            obs_hat, trans_feat_full = self.predict_next_without_kv_cache(critic_obs[:,:,:45],temporal_mask)

            alpha = 1.0
            # get the context input for the autoregressive model
            obs_inputs = critic_obs[:,:32,:45]
            for i in range(prediction_horizon):
                # action_sample = action[:, 32+i:32+i+1, :]
                # pred_obs_full = torch.cat([obs_hat[:, :, :9],       # First 9 elements of pred_obs
                #                         obs[:,32+i:32+i+1,9:12],               # cmd_tensor
                #                             obs_hat[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
                #                             action_sample
                #                             ], dim=-1)
                temporal_mask = get_subsequent_mask_with_batch_length(context_length+i, obs.device)
                obs_hat, trans_feat = self.predict_next_without_kv_cache(obs_inputs,temporal_mask)
                torques = self.compute_torques(action[:, 32+i:32+i+1, :], obs_hat[:, -1:, 9:21], obs_hat[:, -1:, 21:33])
                obs_hat_for_input = torch.cat([obs_hat[:, -1:, :33],       # First 9 elements of pred_obs
                                     torques,   # Remaining elements of pred_obs (from index 9 to 32)
                                    ], dim=-1)
                obs_inputs = torch.cat([obs_inputs, obs_hat_for_input], dim=1)
                obs_hats.append(obs_hat[:,-1:,:])
                # loss for continuous states
                total_loss += alpha**i * self.mse_loss(obs_hat[:,-1:,:45], critic_obs[:,32+i:32+i+1,:45])
                # loss for contact states
                total_loss += alpha**i * self.bce_with_logits_loss_func(obs_hat[:,-1:,45:], critic_obs[:,32+i:32+i+1,45:])
                # loss for latent states
                total_loss += alpha**i * self.mse_loss(trans_feat[:,-1:,:],trans_feat_full[:,32+i:32+i+1,:])
                # total_loss += alpha**i * self.mse_loss(trans_feat, trans_feat_full[:,33+i:33+i+1,:].detach())

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        obs_hats = torch.cat(obs_hats, dim=1)
        observation_difference = torch.abs(obs_hats - critic_obs[:,-8:,:])
        observation_difference = torch.flatten(observation_difference,0,1).mean(dim=0)
        base_vel_diff = observation_difference[0:3].mean()
        angle_diff = observation_difference[3:6].mean()
        projected_diff = observation_difference[6:9].mean()
        dof_pos_diff = observation_difference[9:21].mean()
        dof_vel_diff = observation_difference[21:33].mean()

        if writer is not None and it > 0:
            self.scheduler.step()
            # writer.add_scalar("autoregresser/reconstruction_loss", reconstruction_loss.item(), it)
            # writer.add_scalar("autoregresser/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("autoregresser/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
        if return_loss:
            return total_loss.detach().cpu()