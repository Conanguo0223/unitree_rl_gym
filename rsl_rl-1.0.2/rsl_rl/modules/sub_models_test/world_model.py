import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuadrupedWorldModel(nn.Module):
    """
    Transformer-based world model for quadruped locomotion.
    
    State components:
    - Base linear velocity (3D)
    - Base angular velocity (3D) 
    - Projected gravity (3D)
    - Joint positions (12 for quadruped)
    - Joint velocities (12)
    - Joint torques (12)
    - Foot contact (4 feet, privileged info)
    
    Action: Joint position targets (12)
    """
    
    def __init__(self, 
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_seq_len=32,
                 num_joints=12,
                 num_feet=4):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_joints = num_joints
        self.num_feet = num_feet
        
        # State dimensions
        self.base_linear_vel_dim = 3
        self.base_angular_vel_dim = 3
        self.projected_gravity_dim = 3
        self.joint_pos_dim = num_joints
        self.joint_vel_dim = num_joints
        self.joint_torque_dim = num_joints
        self.foot_contact_dim = num_feet
        self.action_dim = num_joints
        
        # Total state dimension (without privileged info)
        self.obs_dim = (self.base_linear_vel_dim + self.base_angular_vel_dim + 
                       self.projected_gravity_dim + self.joint_pos_dim + 
                       self.joint_vel_dim + self.joint_torque_dim)
        
        # Total state dimension (with privileged info)
        self.state_dim = self.obs_dim + self.foot_contact_dim
        
        # Input embeddings
        self.state_embedding = nn.Linear(self.state_dim, d_model)
        self.action_embedding = nn.Linear(self.action_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Causal mask for autoregressive generation
        self.register_buffer('causal_mask', self._generate_square_subsequent_mask(max_seq_len))
        
        # Output heads for next state prediction
        self.base_linear_vel_head = nn.Linear(d_model, self.base_linear_vel_dim)
        self.base_angular_vel_head = nn.Linear(d_model, self.base_angular_vel_dim)
        self.projected_gravity_head = nn.Linear(d_model, self.projected_gravity_dim)
        self.joint_pos_head = nn.Linear(d_model, self.joint_pos_dim)
        self.joint_vel_head = nn.Linear(d_model, self.joint_vel_dim)
        self.joint_torque_head = nn.Linear(d_model, self.joint_torque_dim)
        self.foot_contact_head = nn.Linear(d_model, self.foot_contact_dim)
        
        # Optional: Keep reward/done heads for comparison, but recommend separate models
        # self.reward_head = nn.Linear(d_model, 1)  # Comment out for pure dynamics model
        # self.done_head = nn.Linear(d_model, 1)    # Comment out for pure dynamics model
        
        self._init_weights()
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_state_tensor(self, base_linear_vel, base_angular_vel, projected_gravity,
                           joint_positions, joint_velocities, joint_torques, foot_contact):
        """
        Concatenate all state components into a single tensor.
        
        Args:
            base_linear_vel: (batch_size, seq_len, 3)
            base_angular_vel: (batch_size, seq_len, 3)
            projected_gravity: (batch_size, seq_len, 3)
            joint_positions: (batch_size, seq_len, num_joints)
            joint_velocities: (batch_size, seq_len, num_joints)
            joint_torques: (batch_size, seq_len, num_joints)
            foot_contact: (batch_size, seq_len, num_feet)
        
        Returns:
            state: (batch_size, seq_len, state_dim)
        """
        return torch.cat([
            base_linear_vel,
            base_angular_vel,
            projected_gravity,
            joint_positions,
            joint_velocities,
            joint_torques,
            foot_contact
        ], dim=-1)
    
    def forward(self, states, actions, mask=None):
        """
        Forward pass of the world model.
        
        Args:
            states: (batch_size, seq_len, state_dim) - Current states
            actions: (batch_size, seq_len, action_dim) - Actions taken
            mask: (batch_size, seq_len) - Optional padding mask
        
        Returns:
            next_states: Dict of predicted next state components
            reward: (batch_size, seq_len, 1) - Predicted rewards
            done: (batch_size, seq_len, 1) - Predicted done flags
        """
        batch_size, seq_len = states.shape[:2]
        
        # Embed states and actions
        state_emb = self.state_embedding(states)  # (batch_size, seq_len, d_model)
        action_emb = self.action_embedding(actions)  # (batch_size, seq_len, d_model)
        
        # Combine state and action embeddings
        # We can either add them or concatenate and project
        combined_emb = state_emb + action_emb
        
        # Add positional encoding
        combined_emb = self.pos_encoding(combined_emb)
        
        # Get causal mask for sequence length
        seq_mask = self.causal_mask[:seq_len, :seq_len] if seq_len <= self.max_seq_len else self._generate_square_subsequent_mask(seq_len)
        
        # Create key padding mask if provided
        key_padding_mask = ~mask if mask is not None else None
        
        # For decoder, we use the same sequence as both target and memory
        # This allows the model to attend to previous timesteps autoregressively
        output = self.transformer(
            tgt=combined_emb,
            memory=combined_emb,
            tgt_mask=seq_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Predict next state components
        next_base_linear_vel = self.base_linear_vel_head(output)
        next_base_angular_vel = self.base_angular_vel_head(output)
        next_projected_gravity = self.projected_gravity_head(output)
        next_joint_pos = self.joint_pos_head(output)
        next_joint_vel = self.joint_vel_head(output)
        next_joint_torque = self.joint_torque_head(output)
        next_foot_contact = torch.sigmoid(self.foot_contact_head(output))  # Sigmoid for binary contact
        
        # For pure dynamics model, don't predict reward/done
        # reward = self.reward_head(output)
        # done = torch.sigmoid(self.done_head(output))
        
        next_states = {
            'base_linear_velocity': next_base_linear_vel,
            'base_angular_velocity': next_base_angular_vel,
            'projected_gravity': next_projected_gravity,
            'joint_positions': next_joint_pos,
            'joint_velocities': next_joint_vel,
            'joint_torques': next_joint_torque,
            'foot_contact': next_foot_contact
        }
        
        return next_states
    
    def predict_sequence(self, initial_state, actions, mask=None):
        """
        Predict a sequence of states given initial state and action sequence.
        
        Args:
            initial_state: (batch_size, 1, state_dim) - Initial state
            actions: (batch_size, seq_len, action_dim) - Action sequence
            mask: (batch_size, seq_len+1) - Optional mask
        
        Returns:
            predicted_states: (batch_size, seq_len+1, state_dim)
        """
        batch_size, seq_len = actions.shape[:2]
        device = actions.device
        
        # Initialize sequence with initial state
        states = [initial_state]
        
        current_state = initial_state
        
        for t in range(seq_len):
            # Get action for current timestep
            current_action = actions[:, t:t+1, :]  # (batch_size, 1, action_dim)
            
            # Predict next state
            next_states = self.forward(
                current_state, 
                current_action,
                mask=mask[:, t:t+1] if mask is not None else None
            )
            
            # Concatenate next state components
            next_state = self.create_state_tensor(
                next_states['base_linear_velocity'],
                next_states['base_angular_velocity'],
                next_states['projected_gravity'],
                next_states['joint_positions'],
                next_states['joint_velocities'],
                next_states['joint_torques'],
                next_states['foot_contact']
            )
            
            states.append(next_state)
            
            # Update current state for next iteration
            current_state = next_state
        
        predicted_states = torch.cat(states, dim=1)  # (batch_size, seq_len+1, state_dim)
        
        return predicted_states


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class QuadrupedRewardModel(nn.Module):
    """
    Separate reward model that takes predicted states and computes rewards.
    This allows for modular design and easy reward function changes.
    """
    
    def __init__(self, state_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, states):
        """
        Compute rewards from states.
        
        Args:
            states: (batch_size, seq_len, state_dim)
        
        Returns:
            rewards: (batch_size, seq_len, 1)
        """
        return self.network(states)


class QuadrupedTerminationModel(nn.Module):
    """
    Separate termination model for episode endings.
    """
    
    def __init__(self, state_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, states):
        """
        Compute termination probabilities from states.
        
        Args:
            states: (batch_size, seq_len, state_dim)
        
        Returns:
            terminations: (batch_size, seq_len, 1)
        """
        return self.network(states)


# Example usage and training loop
def create_sample_data(batch_size=32, seq_len=32, num_joints=12, num_feet=4):
    """Create sample data for testing"""
    
    # Sample states
    base_linear_vel = torch.randn(batch_size, seq_len, 3)
    base_angular_vel = torch.randn(batch_size, seq_len, 3)
    projected_gravity = torch.randn(batch_size, seq_len, 3)
    joint_positions = torch.randn(batch_size, seq_len, num_joints)
    joint_velocities = torch.randn(batch_size, seq_len, num_joints)
    joint_torques = torch.randn(batch_size, seq_len, num_joints)
    foot_contact = torch.randint(0, 2, (batch_size, seq_len, num_feet)).float()
    
    # Sample actions
    actions = torch.randn(batch_size, seq_len, num_joints)
    
    # Create state tensor
    states = torch.cat([
        base_linear_vel, base_angular_vel, projected_gravity,
        joint_positions, joint_velocities, joint_torques, foot_contact
    ], dim=-1)
    
    return states, actions


def train_world_model_modular():
    """Example training loop with separate world model and reward model"""
    
    # Initialize models
    world_model = QuadrupedWorldModel(
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=32
    )
    
    # Separate reward and termination models
    state_dim = world_model.state_dim
    reward_model = QuadrupedRewardModel(state_dim)
    termination_model = QuadrupedTerminationModel(state_dim)
    
    # Loss and optimizers
    dynamics_criterion = nn.MSELoss()
    reward_criterion = nn.MSELoss()
    termination_criterion = nn.BCELoss()
    
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)
    reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    termination_optimizer = torch.optim.Adam(termination_model.parameters(), lr=1e-4)
    
    # Sample training data
    states, actions = create_sample_data()
    target_rewards = torch.randn(states.shape[0], states.shape[1] - 1, 1)  # Sample rewards
    target_terminations = torch.randint(0, 2, (states.shape[0], states.shape[1] - 1, 1)).float()
    
    # Training step for world model
    world_model.train()
    world_optimizer.zero_grad()
    
    # Forward pass through world model
    next_states_pred = world_model(states[:, :-1], actions[:, :-1])
    
    # Target is the next state in sequence
    target_states = states[:, 1:]
    
    # Compute losses for each state component
    losses = {}
    start_idx = 0
    
    for key, pred in next_states_pred.items():
        if key == 'base_linear_velocity':
            end_idx = start_idx + 3
        elif key == 'base_angular_velocity':
            end_idx = start_idx + 3
        elif key == 'projected_gravity':
            end_idx = start_idx + 3
        elif key == 'joint_positions':
            end_idx = start_idx + 12
        elif key == 'joint_velocities':
            end_idx = start_idx + 12
        elif key == 'joint_torques':
            end_idx = start_idx + 12
        elif key == 'foot_contact':
            end_idx = start_idx + 4
        
        target = target_states[:, :, start_idx:end_idx]
        
        if key == 'foot_contact':
            losses[key] = F.binary_cross_entropy(pred, target)
        else:
            losses[key] = dynamics_criterion(pred, target)
        
        start_idx = end_idx
    
    # Total dynamics loss
    dynamics_loss = sum(losses.values())
    dynamics_loss.backward()
    world_optimizer.step()
    
    # Training step for reward model
    reward_model.train()
    reward_optimizer.zero_grad()
    
    predicted_rewards = reward_model(states[:, :-1])
    reward_loss = reward_criterion(predicted_rewards, target_rewards)
    reward_loss.backward()
    reward_optimizer.step()
    
    # Training step for termination model
    termination_model.train()
    termination_optimizer.zero_grad()
    
    predicted_terminations = termination_model(states[:, :-1])
    termination_loss = termination_criterion(predicted_terminations, target_terminations)
    termination_loss.backward()
    termination_optimizer.step()
    
    print(f"Dynamics Loss: {dynamics_loss.item():.4f}")
    print(f"Reward Loss: {reward_loss.item():.4f}")
    print(f"Termination Loss: {termination_loss.item():.4f}")
    
    for key, loss in losses.items():
        print(f"{key} Loss: {loss.item():.4f}")


def evaluate_with_world_model(world_model, reward_model, termination_model, initial_state, actions):
    """
    Example of how to use the modular approach for evaluation/planning.
    """
    world_model.eval()
    reward_model.eval()
    termination_model.eval()
    
    with torch.no_grad():
        # Predict state sequence
        predicted_states = world_model.predict_sequence(initial_state, actions)
        
        # Evaluate rewards and terminations on predicted states
        predicted_rewards = reward_model(predicted_states[:, :-1])  # Exclude initial state
        predicted_terminations = termination_model(predicted_states[:, :-1])
        
        return predicted_states, predicted_rewards, predicted_terminations


if __name__ == "__main__":
    # Test the modular approach
    train_world_model_modular()