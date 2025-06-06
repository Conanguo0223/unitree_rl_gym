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
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, feature_depth, out_bool=False) -> None:
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
        # if self.type == "linear":
        #     backbone.append(
        #         nn.Linear(
        #             in_features=channels,
        #             out_features=original_in_channels,
        #             bias=False
        #         )
        #     )
        # if self.type == "conv":
        #     backbone.append(
        #         nn.ConvTranspose1d(
        #             in_channels=channels,
        #             out_channels=original_in_channels,
        #             kernel_size=4,
        #             stride=2,
        #             padding=1
        #         )
        #     )
        bool_channels = 12
        self.float_head = nn.Linear(channels, original_in_channels-bool_channels)
        self.bool_head = nn.Linear(channels, bool_channels)
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
    
class DistHeadVAE(nn.Module):
    '''
    Dist: abbreviation of distribution for VAE
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, hidden_dim) -> None:
        super().__init__()
        # TODO: add init_noise_std?
        self.stoch_dim = hidden_dim
        # self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        # self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)
        self.mu_in = nn.Linear(image_feat_dim, self.stoch_dim)
        self.log_var_in = nn.Linear(image_feat_dim, self.stoch_dim)

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
        # log_var_in = rearrange(log_var_in, "B L (K C) -> B L K C", K=self.stoch_dim)
        # mu_in = rearrange(mu_in, "B L (K C) -> B L K C", K=self.stoch_dim)
        # logits = self.unimix(logits)
        return mu_in, log_var_in

    def forward_prior_vae(self, x):
        mu_out = self.mu_out(x)
        log_var_out = self.log_var_out(x)
        # log_var_out = rearrange(log_var_out, "B L (K C) -> B L K C", K=self.stoch_dim)
        # mu_out = rearrange(mu_out, "B L (K C) -> B L K C", K=self.stoch_dim)
        # logits = self.prior_head(x)
        # logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        # logits = self.unimix(logits)
        return mu_out, log_var_out


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
class MSELoss_GRU_hid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L F -> B", "sum")
        return loss.mean()
class log_cosh_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = torch.log(torch.cosh(obs_hat - obs))
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
    def __init__(self, in_channels, decoder_out_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.decoder_out_channels = decoder_out_channels
        self.feature_depth = int(math.log2(transformer_hidden_dim) - 5.0) #1:64 #2:128 #3:256
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
            original_in_channels=decoder_out_channels,
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
        # dist = RelaxedOneHotCategorical(logits=logits)
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
            obs_hat_size = (imagine_batch_size, imagine_batch_length+1, self.decoder_out_channels) # set obs dimension
            # other parametes are saying information about the current time step
            action_size = (imagine_batch_size, imagine_batch_length, 12) # set action dimension
            # scalar_size = (imagine_batch_size, imagine_batch_length)
            self.obs_hat_buffer = torch.zeros(obs_hat_size, dtype=dtype, device="cuda")
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")

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
        return last_obs_hat, last_obs_hat, last_reward_hat, last_latent

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
            reward_hat = self.reward_decoder(dist_feat)
            # reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        # self.obs_hat_buffer[:, 0:1] = obs_hat[:, -1:,:]
        self.latent_buffer[:, 0:1] = prior_flattened_sample[:,-1:,:]
        self.hidden_buffer[:, 0:1] = dist_feat[:,-1:,:]
        # return obs_hat[:, -1:,:], prior_flattened_sample[:,-1:,:], dist_feat[:,-1:,:], sample_action[:,-1:,:]# return the predicted observation, the flattend
        return obs_hat[:, -1:,:], prior_flattened_sample[:, -1:].unsqueeze(-1), termination_hat[:, -1:].unsqueeze(-1)

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # modify the obs to not include command and actions
        # which is obs[:, :, 9:12] and obs[:, :, 36:48]
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36], obs[:, :, 48:]], dim=-1)
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36]], dim=-1)
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

            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            flattened_prior_sample = self.flatten_sample(prior_sample)
            
            obs_hat_out = self.image_decoder(flattened_prior_sample)

            # env loss
            # reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs)
            reconstruction_loss = self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,:,:45])
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,:,45:])
            # reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs_decoder)
            reconstruction_loss_out = self.mse_loss_func_obs(obs_hat_out[:,:-1,:], critic_obs[:,1:,:])
            reconstruction_loss_out_contact = self.bce_with_logits_loss_func(obs_hat_out[:,:-1,45:], critic_obs[:,1:,45:])
            
            latent_reconstruction_loss = self.mse_loss_func_obs(flattened_sample[:,:-1,:], flattened_prior_sample[:,1:,:])
            # reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            # reward_loss = self.mse_loss(reward_hat, reward)
            # termination_loss = self.bce_with_logits_loss_func(termination_hat, termination.float())
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            total_loss = reconstruction_loss + reconstruction_loss_contact + reconstruction_loss_out + reconstruction_loss_out_contact + latent_reconstruction_loss + dynamics_loss + representation_loss #+ reward_loss + termination_loss

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
        # contact_diff = observation_difference[33:37].mean()
        # command_diff = observation_difference[9:12].mean()
        # dof_pos_diff = observation_difference[12:24].mean()
        # dof_vel_diff = observation_difference[24:36].mean()
        # dof_action_diff = observation_difference[36:48].mean()
        if writer is not None and it > 0:
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_out", reconstruction_loss_out.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_out_contact", reconstruction_loss_out_contact.item(), it)
            writer.add_scalar("WorldModel/latent_reconstruction_loss", latent_reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_loss", dynamics_loss.item(), it)
            writer.add_scalar("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/representation_loss", representation_loss.item(), it)
            writer.add_scalar("WorldModel/representation_real_kl_div", representation_real_kl_div.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            # writer.add_scalar("obs_diff/command_diff", command_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
            # writer.add_scalar("obs_diff/contact_diff", contact_diff.item(), it)
            # writer.add_scalar("obs_diff/dof_action_diff", dof_action_diff.item(), it)
        if return_loss:
            return reconstruction_loss
    
    def update_tokenizer(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # modify the obs to not include command and actions
        # which is obs[:, :, 9:12] and obs[:, :, 36:48]
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36], obs[:, :, 48:]], dim=-1)
        obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36]], dim=-1)
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

            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            flattened_prior_sample = self.flatten_sample(prior_sample)
            
            obs_hat_out = self.image_decoder(flattened_prior_sample)
            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            # reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs)
            reconstruction_loss = self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,:,:45])
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,:,45:])
            # reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs_decoder)
            reconstruction_loss_out = self.mse_loss_func_obs(obs_hat_out[:,:-1,:], critic_obs[:,1:,:])
            reconstruction_loss_out_contact = self.bce_with_logits_loss_func(obs_hat_out[:,:-1,45:], critic_obs[:,1:,45:])
            # reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            # reward_loss = self.mse_loss(reward_hat, reward)
            # termination_loss = self.bce_with_logits_loss_func(termination_hat, termination.float())
            latent_reconstruction_loss = self.mse_loss_func_obs(flattened_sample[:,:-1,:], flattened_prior_sample[:,1:,:])
            total_loss = reconstruction_loss + reconstruction_loss_contact + reconstruction_loss_out + reconstruction_loss_out_contact + latent_reconstruction_loss#  + reward_loss + termination_loss
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
            writer.add_scalar("WorldModel/tokenizer/reconstruction_loss_out", reconstruction_loss_out.item(), it)
            # writer.add_scalar("WorldModel/tokenizer/reward_loss", reward_loss.item(), it)
            # writer.add_scalar("WorldModel/tokenizer/termination_loss", termination_loss.item(), it)
            writer.add_scalar("WorldModel/tokenizer/total_loss", total_loss.item(), it)


class KLDivLoss(nn.Module):
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

class WorldModel_normal(nn.Module):
    def __init__(self, in_channels, decoder_out_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.decoder_out_channels = decoder_out_channels
        self.feature_depth = int(math.log2(transformer_hidden_dim) - 5.0) #1:64 #2:128 #3:256
        self.stoch_dim = 8
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
        self.dist_head_vae = DistHeadVAE(
            image_feat_dim=self.encoder.last_channels,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=self.stoch_dim
        )
        self.dist_head_contacts = DistHead(
            image_feat_dim=self.encoder.last_channels,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=self.stoch_dim
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=decoder_out_channels,
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
        self.mse_loss = MSELoss()
        self.logcosh_loss = log_cosh_loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        # self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.kl_div_loss = KLDivLoss(free_bits = 1.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=800, gamma=0.1)
    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            mu_post, var_post = self.dist_head_vae.forward_post_vae(embedding)
            flattened_sample = self.reparameterize(mu_post, var_post)
            # flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(last_dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
            # prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)# transformer features
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(dist_feat) # 

            # decoding
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
            # prior_flattened_sample = self.flatten_sample(prior_sample) # the output from transformer to predict the next observation
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample) # predicted next observation
            else:
                obs_hat = None
            

        return obs_hat, mu_prior, var_prior, prior_flattened_sample, dist_feat
    
    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var) + 1e-6
        # Generate random noise using the same shape as std
        q = Normal(mu, std)
        # Return the reparameterized sample
        return q.rsample()
    
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
            obs_hat_size = (imagine_batch_size, imagine_batch_length+1, self.decoder_out_channels) # set obs dimension
            # other parametes are saying information about the current time step
            action_size = (imagine_batch_size, imagine_batch_length, 12) # set action dimension
            # scalar_size = (imagine_batch_size, imagine_batch_length)
            self.obs_hat_buffer = torch.zeros(obs_hat_size, dtype=dtype, device="cuda")
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")
            # self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            # self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

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
        self.obs_hat_buffer[:, 0:1] = last_obs_hat
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            obs_sample = self.obs_hat_buffer[:, i:i+1]
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
        
        last_obs_hat, last_mu_prior, last_var_prior, last_latent, last_dist_feat = self.predict_next(
                context_latent[:,-1:,:],
                sample_action[:, -1:, :],
                log_video=True
            )
        
        # self.obs_hat_buffer[:, 0:1] = obs_hat[:, -1,:]
        # self.latent_buffer[:, 0:1] = prior_flattened_sample[:,-1,:]
        # self.hidden_buffer[:, 0:1] = dist_feat[:,-1,:]
        return last_obs_hat, last_mu_prior, last_var_prior, last_latent

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
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
            # prior_flattened_sample = self.flatten_sample(prior_sample) 
            obs_hat = self.image_decoder(prior_flattened_sample)
            reward_hat = self.reward_decoder(dist_feat)
            # reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0


        # self.obs_hat_buffer[:, 0:1] = obs_hat[:, -1:,:]
        self.latent_buffer[:, 0:1] = prior_flattened_sample[:,-1:,:]
        self.hidden_buffer[:, 0:1] = dist_feat[:,-1:,:]
        
        return obs_hat[:, -1:,:], reward_hat[:, -1:].unsqueeze(-1), termination_hat[:, -1:].unsqueeze(-1) # return the predicted observation, the flattend
    
    def setup_imagination_train(self, batch_size, sample_obs, sample_action, batch_length):
        # if start to step using imagination, initialize the buffer and reset kv_cache
        self.init_imagine_buffer(batch_size, batch_length, dtype=self.tensor_dtype)
        self.storm_transformer.reset_kv_cache_list(batch_size, dtype=self.tensor_dtype)
        self.storm_transformer.train()
        # =====aggregate the kv_cache=====
        # context
        context_latent = self.encode_obs(sample_obs)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(context_latent)
            dist_feat = self.storm_transformer.forward_context(context_latent, sample_action, temporal_mask)
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
            # prior_flattened_sample = self.flatten_sample(prior_sample) 
            obs_hat = self.image_decoder(prior_flattened_sample)
            reward_hat = self.reward_decoder(dist_feat)
            # reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0


        # self.obs_hat_buffer[:, 0:1] = obs_hat[:, -1:,:]
        self.latent_buffer[:, 0:1] = prior_flattened_sample[:,-1:,:]
        self.hidden_buffer[:, 0:1] = dist_feat[:,-1:,:]
        
        return obs_hat[:, -1:,:], prior_flattened_sample[:,-1:,:], mu_prior[:, -1:,:], var_prior[:, -1:,:]# return the predicted observation, the flattend

    def update(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter, return_loss=False):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # modify the obs to not include command and actions
        # which is obs[:, :, 9:12] and obs[:, :, 36:48]
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36], obs[:, :, 48:]], dim=-1)
        # obs_decoder = torch.cat([obs[:, :, :9], obs[:, :, 12:36]], dim=-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            # distribution for float values
            mu_post, var_post = self.dist_head_vae.forward_post_vae(embedding)
            # distribution for discrete values
            flattened_sample = self.reparameterize(mu_post, var_post)
            # flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # transformer
            # TODO: change mask to allowing the first batch_length to all be true
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(dist_feat)
            flattened_prior_sample = self.reparameterize(mu_prior, var_prior)

            obs_hat_out = self.image_decoder(flattened_prior_sample)

            # decoding reward and termination with dist_feat
            # reward_hat = self.reward_decoder(dist_feat)
            # termination_hat = self.termination_decoder(dist_feat)

            # env loss
            # reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs)
            reconstruction_loss = self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,:,:45])
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,:,45:])

            # reconstruction loss on the output
            reconstruction_loss_contact_out = self.bce_with_logits_loss_func(obs_hat_out[:,:-1,45:], critic_obs[:,1:,45:])
            reconstruction_loss_out = self.mse_loss_func_obs(obs_hat_out[:,:-1,:45], critic_obs[:,1:,:45])

            reconstruction_loss = reconstruction_loss + reconstruction_loss_contact
            reconstruction_loss_out = reconstruction_loss_out + reconstruction_loss_contact_out

            latent_reconstruction = self.mse_loss_func_obs(flattened_sample[:,:-1,:], flattened_prior_sample[:,1:,:])
            # reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            # reward_loss = self.mse_loss(torch.unsqueeze(reward_hat,-1), torch.unsqueeze(reward,-1))
            # termination_loss = self.bce_with_logits_loss_func(termination_hat, termination.float())
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(mu_post[:,1:,:].detach(), var_post[:,1:,:].detach(), mu_prior[:,:-1,:], var_prior[:,:-1,:])
            representation_loss, representation_real_kl_div = self.kl_div_loss(mu_post[:,1:,:], var_post[:,1:,:], mu_prior[:,:-1,:].detach(), var_prior[:,:-1,:].detach())

            total_loss = reconstruction_loss + 0.8*reconstruction_loss_out + 0.8*dynamics_loss + 0.8*representation_loss +latent_reconstruction 

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
            writer.add_scalar("WorldModel/latent_reconstruction", latent_reconstruction.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("WorldModel/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("WorldModel/dynamics_loss", dynamics_loss.item(), it)
            # writer.add_scalar("WorldModel/reward_loss", reward_loss.item(), it)
            # writer.add_scalar("WorldModel/termination_loss", termination_loss.item(), it)
            writer.add_scalar("WorldModel/representation_loss", representation_loss.item(), it)
            # writer.add_scalar("WorldModel/real_kl_loss", real_kl_loss.item(), it)
            writer.add_scalar("WorldModel/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
            # writer.add_scalar("obs_diff/contact_diff", contact_diff.item(), it)
            # writer.add_scalar("obs_diff/dof_action_diff", dof_action_diff.item(), it)
        if return_loss:
            return reconstruction_loss
    
    def update_tokenizer(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        # scale reward to be larger

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            mu_post, var_post = self.dist_head_vae.forward_post_vae(embedding)
            flattened_sample = self.reparameterize(mu_post, var_post)
            # flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)
            
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(dist_feat)
            flattened_prior_sample = self.reparameterize(mu_prior, var_prior)
            # env loss
            # reconstruction_loss = self.mse_loss_func_obs(obs_hat, obs)
            latent_reconstruction = self.mse_loss_func_obs(flattened_sample[:,:-1,:], flattened_prior_sample[:,1:,:])
            reconstruction_loss = self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,:,:45])
            reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,:,45:])
            
            total_loss = reconstruction_loss + reconstruction_loss_contact + latent_reconstruction

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if writer is not None and it > 0:
            self.scheduler.step()
            writer.add_scalar("tokenizer/latent_reconstruction", latent_reconstruction.item(), it)
            writer.add_scalar("tokenizer/reconstruction_loss", reconstruction_loss.item(), it)
            writer.add_scalar("tokenizer/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("tokenizer/total_loss", total_loss.item(), it)


    def update_autoregressive(self, obs, critic_obs, action, reward, termination, it, writer: SummaryWriter):
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # pass through the context and get the first prediction
            obs_hat, prior_latent_sample, mu_prior, var_prior= self.setup_imagination_train(obs.shape[0], obs[:,:32,:], action[:,:32,:], 32)
            embedding = self.encoder(obs)
            mu_post, var_post = self.dist_head_vae.forward_post_vae(embedding)
            flattened_sample = self.reparameterize(mu_post, var_post)
            obs_hats = obs_hat
            latent_sampels = prior_latent_sample
            mu_priors = mu_prior
            var_priors = var_prior
            total_loss = 0
            total_loss += self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,32:33,:45])
            total_loss += self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,32:33,45:])
            total_loss += self.mse_loss_func_obs(prior_latent_sample, flattened_sample[:,32:33,:])
            alpha = 0.8
            
            for i in range(7):
                action_sample = action[:, 32+i:32+i+1, :]
                pred_obs_full = torch.cat([obs_hat[:, :, :9],       # First 9 elements of pred_obs
                                        obs[:,32+i:32+i+1,9:12],               # cmd_tensor
                                            obs_hat[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
                                            action_sample
                                            ], dim=-1)
                
                obs_hat, mu_prior, var_prior, prior_latent_sample = self.imagine_step( None, pred_obs_full, action_sample, None, None, None)
                obs_hats = torch.cat([obs_hats, obs_hat], dim=1)
                latent_sampels = torch.cat([latent_sampels, prior_latent_sample], dim=1)
                mu_priors = torch.cat([mu_priors, mu_prior], dim=1)
                var_priors = torch.cat([var_priors, var_prior], dim=1)

                total_loss += alpha**i * self.mse_loss_func_obs(obs_hat[:,:,:45], critic_obs[:,33+i:33+i+1,:45])
                total_loss += alpha**i * self.bce_with_logits_loss_func(obs_hat[:,:,45:], critic_obs[:,33+i:33+i+1,45:])
                total_loss += alpha**i * self.mse_loss_func_obs(latent_sampels, flattened_sample[:,33+i:33+i+1,:])
            
            

            # latent_reconstruction = self.mse_loss_func_obs(latent_sampels, flattened_sample[:,-8:,:])
            # reconstruction_loss = self.mse_loss_func_obs(obs_hats[:,:,:45], critic_obs[:,-8:,:45])
            # latent_reconstruction = self.logcosh_loss(latent_sampels, flattened_sample[:,-8:,:])
            # reconstruction_loss = self.logcosh_loss(obs_hats[:,:,:45], critic_obs[:,-8:,:45])
            # reconstruction_loss_contact = self.bce_with_logits_loss_func(obs_hats[:,:,45:], critic_obs[:,-8:,45:])
            dynamics_loss, dynamics_real_kl_div = self.kl_div_loss(mu_post[:,-8:,:].detach(), var_post[:,-8:,:].detach(), mu_priors[:,:,:], var_priors[:,:,:])
            representation_loss, representation_real_kl_div = self.kl_div_loss(mu_post[:,-8:,:], var_post[:,-8:,:], mu_priors[:,:,:].detach(), var_priors[:,:,:].detach())
            
            
            total_loss = total_loss + 0.5 * dynamics_loss+ 0.1 * representation_loss
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
            # writer.add_scalar("autoregresser/reconstruction_loss", reconstruction_loss.item(), it)
            # writer.add_scalar("autoregresser/reconstruction_loss_contact", reconstruction_loss_contact.item(), it)
            writer.add_scalar("autoregresser/total_loss", total_loss.item(), it)
            writer.add_scalar("obs_diff/base_vel_diff", base_vel_diff.item(), it)
            writer.add_scalar("obs_diff/angle_diff", angle_diff.item(), it)
            writer.add_scalar("obs_diff/projected_diff", projected_diff.item(), it)
            writer.add_scalar("obs_diff/dof_pos_diff", dof_pos_diff.item(), it)
            writer.add_scalar("obs_diff/dof_vel_diff", dof_vel_diff.item(), it)
            writer.add_scalar("autoregresser/dynamics_loss", dynamics_loss.item(), it)
            writer.add_scalar("autoregresser/representation_loss", representation_loss.item(), it)
            writer.add_scalar("autoregresser/dynamics_real_kl_div", dynamics_real_kl_div.item(), it)
            writer.add_scalar("autoregresser/representation_real_kl_div", representation_real_kl_div.item(), it)

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
        # pred_seq = []
        # input_t = obs_sequence[:,-1:,:]
        
        # for _ in range(prediction_horizon):
        #     action_sample = action_sequence[:, -1:, :] # (batch, 1, action_dim)
        #     command_sample = obs_sequence[:, -1:, 9:12] # (batch, 1, 3)
        #     obs_hat,h = self.gru(input_t, h) # out: (batch, 1, hidden_dim)
        #     obs_hat = self.obs_head(obs_hat) # obs_hat: (batch, 1, input_dim)
        #     mu_o, logvar_o = torch.chunk(obs_hat, 2, dim=-1)
        #     obs_hat = self.reparameterize(mu_o, logvar_o)
        #     pred_seq.append(obs_hat)
        #     input_t = torch.cat([obs_hat[:, :, :9],       # First 9 elements of pred_obs
        #                          command_sample,               # cmd_tensor
        #                          obs_hat[:, :, 9:33],     # Remaining elements of pred_obs (from index 9 to 32)
        #                          action_sample
        #                         ], dim=-1)
        #      # use prediction as next input

        # return torch.cat(pred_seq, dim=1)     # (batch, future_steps, input_dim)
    
    def compute_torques(self, actions, dof_pos_diff, dof_vel):
        actions_scaled = actions * 0.25
        # dof_pos_diff = -1*obs[9:21]/obs_scales.dof_pos
        # dof_vel = obs[21:33]/obs_scales.dof_vel
        dof_pos_diff = dof_pos_diff / 1.0
        dof_vel = dof_vel / 0.05
        torques = 20*(actions_scaled - dof_pos_diff) - 0.5*dof_vel
        return torques

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
                torques = self.compute_torques(action_sample, obs_hat[:, :, 9:21], obs_hat[:, :, 21:33])
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
    def __init__(self, in_channels, decoder_out_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads,
                 lr = 1e-3):
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
        self.kl_div_loss = KLDivLoss(free_bits = 1.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(last_dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
            # prior_flattened_sample = self.flatten_sample(prior_sample)
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

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

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
        self.kl_div_loss = KLDivLoss(free_bits = 1.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            mu_prior, var_prior = self.dist_head_vae.forward_prior_vae(last_dist_feat)
            prior_flattened_sample = self.reparameterize(mu_prior, var_prior)
            # prior_flattened_sample = self.flatten_sample(prior_sample)
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

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

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

