import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from .functions_losses import SymLogTwoHotLoss
from .attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from .transformer_model import StochasticTransformerKVCache
from .agents import ActorCriticAgent


class EncoderBN(nn.Module):
    def __init__(self, in_features, stem_channels, feature_depth) -> None:
        super().__init__()
        # stem_channels = 32
        # final_feature_layers = 4
        # self.type = "conv"
        self.type = "linear"
        backbone = []
        # stem
        if self.type == "linear":
            backbone.append(
                nn.Linear(
                    in_features= in_features, 
                    out_features=stem_channels,
                    bias=False
                )
        )
        if self.type == "conv":
            backbone.append(
                nn.Conv1d(
                    in_channels=in_features,
                    out_channels=stem_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
        feature_layers = feature_depth
        channels = stem_channels
        backbone.append(nn.BatchNorm1d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            if self.type == "linear":
                backbone.append(
                    nn.Linear(
                        in_features=channels,
                        out_features=channels*2,
                        bias=False
                    )
                )
            if self.type == "conv":
                backbone.append(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels*2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
            channels *= 2
            feature_layers -= 1
            backbone.append(nn.BatchNorm1d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_layers == 0:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B T F -> (B T) F")
        x = self.backbone(x)
        x = rearrange(x, "(B T) F -> B T F", B=batch_size)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """
        Vector Quantizer for VQ-VAE.

        Args:
            num_embeddings (int): Number of embedding vectors in the codebook.
            embedding_dim (int): Dimensionality of each embedding vector.
            commitment_cost (float): Weight for the commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize the embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        """
        Forward pass for vector quantization.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, embedding_dim, ...).

        Returns:
            quantized (torch.Tensor): Quantized tensor of the same shape as inputs.
            loss (torch.Tensor): Commitment loss.
            indices (torch.Tensor): Indices of the closest embeddings.
        """
        # Flatten input to (batch_size * ..., embedding_dim)
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Compute distances to embedding vectors
        distances = (
            (flat_inputs ** 2).sum(dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.embedding.weight.T)
            + (self.embedding.weight ** 2).sum(dim=1)
        )

        # Find the closest embedding for each input
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view_as(inputs)

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through gradient estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, indices

class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, feature_depth) -> None:
        super().__init__()
        self.type = "linear"
        # self.type = "conv"
        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels, bias=False))
        backbone.append(Rearrange('B L F -> (B L) F', F=last_channels))
        backbone.append(nn.BatchNorm1d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = feature_depth
        while True:
            if channels == stem_channels:
                break
            if self.type == "linear":
                backbone.append(
                    nn.Linear(
                        in_features=channels,
                        out_features=channels//2,
                        bias=False
                    )
                )
            if self.type == "conv":
                backbone.append(
                    nn.ConvTranspose1d(
                        in_channels=channels,
                        out_channels=channels//2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm1d(channels))
            backbone.append(nn.ReLU(inplace=True))
        if self.type == "linear":
            backbone.append(
                nn.Linear(
                    in_features=channels,
                    out_features=original_in_channels,
                    bias=False
                )
            )
        if self.type == "conv":
            backbone.append(
                nn.ConvTranspose1d(
                    in_channels=channels,
                    out_channels=original_in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0] # B L (K C)
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) F -> B L F", B=batch_size)
        return obs_hat


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
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
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
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
            # nn.Sigmoid()
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


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.feature_depth = 4
        self.stoch_dim = 32
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.in_channels = in_channels
        self.encoder = EncoderBN(
            in_features=in_channels,
            stem_channels=32,
            feature_depth=self.feature_depth
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1
        )
        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=self.stoch_dim
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=in_channels,
            stem_channels=32,
            feature_depth=self.feature_depth
        )
        self.reward_decoder = RewardDecoder(
            num_classes=1,
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )

        self.mse_loss_func_obs = MSELoss()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        # self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            prior_logits = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)# transformer features
            prior_logits = self.dist_head.forward_prior(dist_feat) # 

            # decoding
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample) # the output from transformer to predict the next observation
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample) # predicted next observation
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(dist_feat)
            # reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

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
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            obs_hat_size = (imagine_batch_size, imagine_batch_length+1, self.in_channels) # set obs dimension
            # other parametes are saying information about the current time step
            action_size = (imagine_batch_size, imagine_batch_length, 12) # set action dimension
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.obs_hat_buffer = torch.zeros(obs_hat_size, dtype=dtype, device="cuda")
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def imagine_data(self, agent: ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)
        obs_hat_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

        if log_video:
            logger.log("Imagine/predict_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())

        return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer
    
    def imagine_step(self, batch_size, sample_obs, sample_action, reward_sample, termination_sample, imag_step):
        # context
        # uset the encoder to encode the obs
        context_latent = self.encode_obs(sample_obs[:,-1:,:])
        
        last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:,-1:,:],
                sample_action[:, -1:, :],
                log_video=True
            )
        
        # self.obs_hat_buffer[:, 0:1] = obs_hat[:, -1,:]
        # self.latent_buffer[:, 0:1] = prior_flattened_sample[:,-1,:]
        # self.hidden_buffer[:, 0:1] = dist_feat[:,-1,:]
        return last_obs_hat, last_obs_hat, last_reward_hat, last_termination_hat

    def setup_imagination(self, batch_size, sample_obs, sample_action, batch_length):
        # if start to step using imagination, initialize the buffer and reset kv_cache
        self.init_imagine_buffer(batch_size, batch_length, dtype=self.tensor_dtype)
        self.storm_transformer.reset_kv_cache_list(batch_size, dtype=self.tensor_dtype)
        self.storm_transformer.eval()
        # =====aggregate the kv_cache=====
        # context
        context_latent = self.encode_obs(sample_obs)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(context_latent)
            dist_feat = self.storm_transformer.forward_context(context_latent, sample_action, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample) 
            obs_hat = self.image_decoder(prior_flattened_sample)

        self.obs_hat_buffer[:, 0:1] = obs_hat[:, -1:,:]
        self.latent_buffer[:, 0:1] = prior_flattened_sample[:,-1:,:]
        self.hidden_buffer[:, 0:1] = dist_feat[:,-1:,:]
        return obs_hat[:, -1:,:], prior_flattened_sample[:,-1:,:], dist_feat[:,-1:,:]# return the predicted observation, the flattend

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs)
            # reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            reward_loss = self.mse_loss(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination.float())
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if writer is not None and it > 0:
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reward_loss", reward_loss.item(), it)
            writer.add_scalar("WorldModel/termination_loss", termination_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_loss", dynamics_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/representation_loss", representation_loss.item(), it)
            writer.add_scalar("WorldModel/representation_real_kl_div", representation_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
    
    def update_tokenizer(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs)
            # reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            reward_loss = self.mse_loss(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination.float())
            total_loss = reconstruction_loss + reward_loss + termination_loss
            # # dyn-rep loss
            # dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            # representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            # total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if writer is not None and it > 0:
            writer.add_scalar("WorldModel/tokenizer/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/tokenizer/reward_loss", reward_loss.item(), it)
            writer.add_scalar("WorldModel/tokenizer/termination_loss", termination_loss.item(), it)
            writer.add_scalar("WorldModel/tokenizer/total_loss", total_loss.item(), it)
