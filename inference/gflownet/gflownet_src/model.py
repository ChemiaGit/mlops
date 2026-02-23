#!/usr/bin/env python3
"""
GFlowNet Model for Faujasite Structure Generation

Implements Trajectory Balance (TB) GFlowNet for generating diverse
Faujasite structures with targeted CO2 adsorption properties.

Reference: Bengio et al. "Flow Network based Generative Models for Non-Iterative
Diverse Candidate Generation" (NeurIPS 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from src.gflownet.environment import FaujasiteEnvironment, FaujasiteState


@dataclass
class Trajectory:
    """A complete trajectory from s0 to terminal state"""
    states: List[FaujasiteState]
    actions: List[int]
    rewards: List[float]  # Can be intermediate rewards
    final_reward: float  # R(x) from oracle
    sum_log_prob: Optional[float] = None  # Sum of log probs under sampling policy (for importance weighting)
    terminal_log_prob: float = 0.0  # log P(STOP | final_state) under sampling policy


class ReplayBuffer:
    """
    Prioritized replay buffer for off-policy GFlowNet training.

    Stores high-reward trajectories for replay alongside fresh on-policy samples.
    Priority is reward^alpha, so higher-reward trajectories are replayed more often.
    """

    def __init__(self, max_size: int = 1000, alpha: float = 1.0):
        """
        Args:
            max_size: Maximum number of trajectories to store
            alpha: Priority exponent (higher = more biased toward high reward)
        """
        self.max_size = max_size
        self.alpha = alpha
        self.buffer: List[Trajectory] = []
        self.priorities: List[float] = []

    def add(self, trajectories: List[Trajectory]):
        """Add trajectories to buffer, evicting lowest-priority if full."""
        for traj in trajectories:
            priority = max(traj.final_reward, 1e-8) ** self.alpha
            if len(self.buffer) < self.max_size:
                self.buffer.append(traj)
                self.priorities.append(priority)
            else:
                # Replace lowest priority entry
                min_idx = int(np.argmin(self.priorities))
                if priority > self.priorities[min_idx]:
                    self.buffer[min_idx] = traj
                    self.priorities[min_idx] = priority

    def sample(self, n: int, strategy: str = 'prioritized') -> List[Trajectory]:
        """
        Sample trajectories from buffer.

        Args:
            n: Number of trajectories to sample
            strategy: 'prioritized' (weighted by priority) or 'uniform'

        Returns:
            List of sampled trajectories
        """
        if len(self.buffer) == 0:
            return []

        n = min(n, len(self.buffer))

        if strategy == 'prioritized':
            probs = np.array(self.priorities)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
        else:
            indices = random.sample(range(len(self.buffer)), n)

        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    """
    Policy network for GFlowNet forward policy.

    Maps state → action logits (probabilities of taking each action)
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()

        layers = []
        in_dim = state_dim

        # Build MLP
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output layer: action logits (small init for near-uniform initial policy)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, state_tensor: torch.Tensor, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.

        Args:
            state_tensor: (batch_size, state_dim)
            valid_actions_mask: (batch_size, num_actions) - 1 for valid, 0 for invalid

        Returns:
            action_logits: (batch_size, num_actions)
        """
        features = self.encoder(state_tensor)
        logits = self.action_head(features)

        # Mask invalid actions (set to -inf)
        logits = logits.masked_fill(valid_actions_mask == 0, float('-inf'))

        return logits


class FlowNetwork(nn.Module):
    """
    Flow network: estimates log Z (partition function).

    Supports both a simple learned scalar (state_dim=0) and a
    state-conditioned estimator that adjusts log Z based on initial state features.
    """

    def __init__(self, state_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        # Learnable base log partition function
        self.log_Z_base = nn.Parameter(torch.zeros(1))

        # Optional state-conditioned adjustment
        if state_dim > 0:
            self.state_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.state_net = None

    def forward(self, state_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log Z, optionally conditioned on initial state features.

        Args:
            state_features: (batch_size, state_dim) or None

        Returns:
            log_Z: scalar or (batch_size, 1) tensor
        """
        if self.state_net is not None and state_features is not None:
            return self.log_Z_base + self.state_net(state_features)
        return self.log_Z_base


class GFlowNet:
    """
    GFlowNet for Faujasite structure generation.

    Uses Trajectory Balance (TB) objective for training.
    """

    def __init__(
        self,
        env: FaujasiteEnvironment,
        hidden_dim: int = 256,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        flow_lr_multiplier: float = 100.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.device = device

        # Initialize networks
        state_dim = env.get_state_dim()
        num_actions = env.num_t_sites + 1  # T-sites + terminal action
        self.policy_net = PolicyNetwork(state_dim, num_actions, hidden_dim, num_layers).to(device)
        self.flow_net = FlowNetwork(state_dim=state_dim, hidden_dim=hidden_dim // 4).to(device)

        # Separate learning rates: flow network needs much higher lr than policy
        # because Adam normalizes gradients, so log Z moves ~lr per step regardless
        # of loss magnitude. Standard practice in GFlowNet literature.
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': learning_rate},
            {'params': self.flow_net.parameters(), 'lr': learning_rate * flow_lr_multiplier},
        ])

        print(f"GFlowNet initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Device: {device}")
        print(f"  Policy lr: {learning_rate}")
        print(f"  Flow lr: {learning_rate * flow_lr_multiplier} ({flow_lr_multiplier}x)")
        print(f"  Parameters: {self.count_parameters():,}")

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.policy_net.parameters()) + \
               sum(p.numel() for p in self.flow_net.parameters())

    def get_action_logits(self, state: FaujasiteState) -> torch.Tensor:
        """Get action logits for a single state"""
        state_tensor = torch.FloatTensor(self.env.state_to_tensor(state)).unsqueeze(0).to(self.device)

        # Create valid actions mask
        valid_actions = self.env.get_valid_actions(state)
        max_actions = self.env.num_t_sites + 1  # +1 for terminal
        mask = torch.zeros(1, max_actions)
        mask[0, valid_actions] = 1
        mask = mask.to(self.device)

        with torch.no_grad():
            logits = self.policy_net(state_tensor, mask)

        return logits[0]

    def sample_action(
        self, state: FaujasiteState, temperature: float = 1.0,
        epsilon: float = 0.0, noise_scale: float = 0.0
    ) -> Tuple[int, float]:
        """
        Sample action from policy with epsilon-uniform exploration and optional noise.

        Args:
            state: Current state
            temperature: Sampling temperature (1.0 = sample from policy, 0.0 = greedy)
            epsilon: Exploration rate. Blends policy with uniform over valid actions.
                     0.0 = pure policy, 1.0 = pure uniform.
            noise_scale: Scale of Gaussian noise added to logits before softmax.
                         Preserves policy structure while adding stochasticity.

        Returns:
            (action, log_prob) - log_prob is always under the pure policy (for TB loss)
        """
        logits = self.get_action_logits(state)

        # Apply temperature
        logits = logits / temperature

        # Add Gaussian noise to valid action logits only
        if noise_scale > 0:
            valid_mask = (logits > float('-inf'))
            noise = noise_scale * torch.randn_like(logits)
            logits = logits + noise * valid_mask.float()

        # Policy probabilities
        probs = F.softmax(logits, dim=0)

        # Epsilon-uniform exploration: blend with uniform over valid actions
        if epsilon > 0:
            valid_mask = (logits > float('-inf')).float()
            num_valid = valid_mask.sum()
            if num_valid > 0:
                uniform_probs = valid_mask / num_valid
                probs = (1 - epsilon) * probs + epsilon * uniform_probs

        action = torch.multinomial(probs, 1).item()

        # Log prob under pure policy (not exploration policy) for TB loss
        log_prob = F.log_softmax(logits, dim=0)[action].item()

        return action, log_prob

    def sample_trajectory(
        self, temperature: float = 1.0, epsilon: float = 0.0,
        noise_scale: float = 0.0
    ) -> Tuple[List[FaujasiteState], List[int], List[float], float]:
        """
        Sample a complete trajectory using the current policy.

        Args:
            temperature: Sampling temperature
            epsilon: Exploration rate for epsilon-uniform blending
            noise_scale: Gaussian noise scale for logit perturbation

        Returns:
            (states, actions, log_probs, terminal_log_prob)
        """
        states = [self.env.reset()]
        actions = []
        log_probs = []
        terminal_log_prob = 0.0  # Default: forced termination (log P = 0)

        while not self.env.is_terminal(states[-1]):
            state = states[-1]
            action, log_prob = self.sample_action(state, temperature, epsilon, noise_scale)

            # Check if terminal action
            if action >= self.env.num_t_sites:
                terminal_log_prob = log_prob  # Record P(STOP | state)
                break

            next_state = self.env.step(state, action)

            states.append(next_state)
            actions.append(action)
            log_probs.append(log_prob)

            # Safety check
            if len(actions) > 100:
                print("Warning: Trajectory exceeded 100 steps")
                break

        return states, actions, log_probs, terminal_log_prob

    def compute_tb_loss(
        self, trajectories: List[Trajectory],
        importance_clip: float = 5.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Trajectory Balance loss with optional importance weighting.

        TB loss: (log Z + log P_F) - (log R + log P_B)

        For off-policy (replay) trajectories, applies importance weight:
            w = min(clip, P_current / P_old) to the squared loss.

        Args:
            trajectories: List of trajectories with rewards
            importance_clip: Max importance weight for stability

        Returns:
            (loss, info_dict)
        """
        losses = []

        for traj in trajectories:
            # Get initial state features for state-conditioned log Z
            initial_state = traj.states[0]
            initial_features = torch.FloatTensor(
                self.env.state_to_tensor(initial_state)
            ).unsqueeze(0).to(self.device)

            # Forward log probability: sum of log P(a|s)
            log_pf = 0.0
            for state, action in zip(traj.states[:-1], traj.actions):
                state_tensor = torch.FloatTensor(self.env.state_to_tensor(state)).unsqueeze(0).to(self.device)

                # Valid actions mask
                valid_actions = self.env.get_valid_actions(state)
                max_actions = self.env.num_t_sites + 1
                mask = torch.zeros(1, max_actions).to(self.device)
                mask[0, valid_actions] = 1

                logits = self.policy_net(state_tensor, mask)
                log_probs = F.log_softmax(logits, dim=1)
                log_pf += log_probs[0, action]

            # Terminal action probability under current policy
            final_state = traj.states[-1]
            if not self.env.is_terminal(final_state):
                # Model chose to terminate — compute log P(STOP | final_state)
                state_tensor = torch.FloatTensor(
                    self.env.state_to_tensor(final_state)
                ).unsqueeze(0).to(self.device)
                valid_actions = self.env.get_valid_actions(final_state)
                max_actions = self.env.num_t_sites + 1
                mask = torch.zeros(1, max_actions).to(self.device)
                mask[0, valid_actions] = 1
                logits = self.policy_net(state_tensor, mask)
                log_probs_terminal = F.log_softmax(logits, dim=1)
                terminal_action = self.env.num_t_sites  # STOP action index
                log_pf += log_probs_terminal[0, terminal_action]

            # Backward log probability: uniform backward policy
            # At each backward step t, we remove one of t Al atoms with prob 1/t
            # So log P_B = -sum(log(t) for t in 1..T) = -log(T!)
            T = len(traj.actions)
            log_pb = -sum(np.log(t) for t in range(1, T + 1)) if T > 0 else 0.0

            # Reward
            reward = max(traj.final_reward, 1e-8)  # Avoid log(0)
            log_reward = torch.log(torch.tensor(reward)).to(self.device)

            # Flow (partition function) - state-conditioned
            log_Z = self.flow_net(initial_features).squeeze()

            # TB loss: (log Z + log P_F) - (log R + log P_B)
            loss = (log_Z + log_pf) - (log_reward + log_pb)
            squared_loss = loss ** 2

            # Importance weighting for off-policy trajectories
            if traj.sum_log_prob is not None:
                current_log_prob = log_pf.detach().item() if torch.is_tensor(log_pf) else log_pf
                log_ratio = current_log_prob - traj.sum_log_prob
                importance_weight = min(importance_clip, np.exp(log_ratio))
                squared_loss = squared_loss * importance_weight

            losses.append(squared_loss)

        total_loss = torch.stack(losses).mean()

        info = {
            'loss': total_loss.item(),
            'log_Z': self.flow_net().item(),
            'mean_reward': np.mean([t.final_reward for t in trajectories]),
            'max_reward': np.max([t.final_reward for t in trajectories]),
        }

        return total_loss, info

    def train_step(
        self, trajectories: List[Trajectory],
        replay_trajectories: Optional[List[Trajectory]] = None
    ) -> Dict:
        """
        Single training step.

        Args:
            trajectories: Batch of fresh on-policy trajectories with rewards
            replay_trajectories: Optional off-policy trajectories from replay buffer

        Returns:
            info_dict with loss and metrics
        """
        # Concatenate fresh + replay trajectories
        all_trajectories = list(trajectories)
        if replay_trajectories:
            all_trajectories.extend(replay_trajectories)

        self.optimizer.zero_grad()

        loss, info = self.compute_tb_loss(all_trajectories)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return info

    def initialize_log_z(self, fast_proxy, num_samples: int = 100):
        """
        Initialize log Z using fast proxy evaluations of random configurations.

        Samples random configs, evaluates with fast proxy, and sets
        log_Z_base = log(mean(rewards)) for better initial convergence.

        Args:
            fast_proxy: FastPhysicsProxy instance with evaluate(config) method
            num_samples: Number of random configs to sample
        """
        rewards = []
        for _ in range(num_samples):
            states, _, _, _ = self.sample_trajectory(temperature=2.0)
            final_state = states[-1]
            result = fast_proxy.evaluate(final_state.config)
            rewards.append(max(result['reward'], 1e-8))

        mean_reward = np.mean(rewards)
        init_log_z = float(np.log(mean_reward))

        with torch.no_grad():
            self.flow_net.log_Z_base.fill_(init_log_z)

        print(f"  Initialized log Z = {init_log_z:.3f} (from {num_samples} random samples, mean_reward={mean_reward:.4f})")

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'flow_net': self.flow_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        # Handle old checkpoints with scalar log_Z
        try:
            self.flow_net.load_state_dict(checkpoint['flow_net'])
        except RuntimeError:
            # Old checkpoint had log_Z, new has log_Z_base + state_net
            old_state = checkpoint['flow_net']
            if 'log_Z' in old_state and 'log_Z_base' not in old_state:
                old_state['log_Z_base'] = old_state.pop('log_Z')
            self.flow_net.load_state_dict(old_state, strict=False)
            print(f"  Note: Loaded legacy checkpoint, state_net initialized fresh")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded from {path}")


def test_gflownet():
    """Test GFlowNet implementation"""
    print("\n" + "="*70)
    print("Testing GFlowNet")
    print("="*70)

    # Create environment
    env = FaujasiteEnvironment(max_substitutions=5)

    # Create GFlowNet
    gfn = GFlowNet(env, hidden_dim=128, num_layers=2)

    # Sample trajectory
    print("\nSampling trajectory...")
    states, actions, log_probs, terminal_lp = gfn.sample_trajectory(temperature=1.0)
    print(f"  Trajectory length: {len(actions)}")
    print(f"  Actions: {actions}")
    print(f"  Terminal log prob: {terminal_lp:.4f}")
    print(f"  Final Si/Al ratio: {env.get_si_al_ratio(states[-1]):.2f}")

    # Sample with noise
    print("\nSampling with noise_scale=0.5...")
    states_n, actions_n, _, _ = gfn.sample_trajectory(temperature=1.0, noise_scale=0.5)
    print(f"  Trajectory length: {len(actions_n)}")

    # Create dummy trajectories with rewards
    print("\nTesting training step...")
    trajectories = []
    for i in range(4):
        states, actions, log_probs, terminal_lp = gfn.sample_trajectory()
        traj = Trajectory(
            states=states,
            actions=actions,
            rewards=[0.0] * len(actions),
            final_reward=np.random.uniform(0.1, 1.0),
            sum_log_prob=sum(log_probs) + terminal_lp,
            terminal_log_prob=terminal_lp,
        )
        trajectories.append(traj)

    info = gfn.train_step(trajectories)
    print(f"  Loss: {info['loss']:.4f}")
    print(f"  log(Z): {info['log_Z']:.4f}")

    # Test replay buffer
    print("\nTesting ReplayBuffer...")
    replay = ReplayBuffer(max_size=10, alpha=1.0)
    replay.add(trajectories)
    print(f"  Buffer size: {len(replay)}")
    sampled = replay.sample(2, strategy='prioritized')
    print(f"  Sampled {len(sampled)} trajectories")

    # Test training with replay
    print("\nTesting train_step with replay...")
    info = gfn.train_step(trajectories[:2], replay_trajectories=sampled)
    print(f"  Loss: {info['loss']:.4f}")

    print("\n" + "="*70)
    print("GFlowNet test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_gflownet()
