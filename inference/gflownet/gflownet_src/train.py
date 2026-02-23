#!/usr/bin/env python3
"""
Training Script for GFlowNet-Based Faujasite Discovery

This script implements the full training loop:
1. GFlowNet generates Faujasite structures
2. Oracle evaluates CO2 adsorption energy
3. Reward signal updates GFlowNet policy
4. Repeat until convergence

Usage:
    python train_gflownet.py --num_iterations 1000 --batch_size 16
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np

import torch

from src.gflownet.environment import FaujasiteEnvironment, FaujasiteState
from src.gflownet.model import GFlowNet, Trajectory, ReplayBuffer
from src.gflownet.fairchem_proxy import FAIRChemProxy
from src.gflownet.fast_proxy import FastPhysicsProxy, ProxyConfig, QuadraticEnergyModel
from src.oracle.fairchem_oracle import FAIRChemOracle
from src.evaluation.batched_evaluator import BatchedEvaluator, BFGSRelaxedEvaluator, GPUSequentialEvaluator


class GFlowNetTrainer:
    """Trainer for GFlowNet with oracle evaluation"""

    def __init__(
        self,
        env: FaujasiteEnvironment,
        gfn: GFlowNet,
        oracle: FAIRChemOracle,
        output_dir: str = "gfn_results",
        use_mock_oracle: bool = False,
        use_fast_proxy: bool = False,
        use_bfgs: bool = True,
        use_batching: bool = True,
        use_gpu: bool = True,
        num_workers: int = None,
        fmax: float = 0.05,
        max_bfgs_steps: int = 20,
        num_initial_positions: int = 1,
        replay_buffer_size: int = 1000,
        replay_alpha: float = 1.0,
        oversample_factor: int = 1,
    ):
        self.env = env
        self.gfn = gfn
        self.oracle = oracle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_mock_oracle = use_mock_oracle
        self.use_fast_proxy = use_fast_proxy
        self.use_bfgs = use_bfgs
        self.use_batching = use_batching
        self.use_gpu = use_gpu

        # Initialize evaluators
        target_energy = oracle.target_energy if hasattr(oracle, 'target_energy') else -0.5

        # Energy model aligned with target: quadratic minimum = target_energy
        # so that optimal configs naturally produce energies near the reward peak.
        # Default QuadraticEnergyModel has min_energy=-0.25 which is misaligned with
        # the -0.5 eV target, causing most structures to get zero reward.
        self._aligned_energy_model = QuadraticEnergyModel(
            optimal_al_frac=0.15,
            min_energy=target_energy,  # Align with reward target
            zero_al_energy=0.3,
        )

        if use_fast_proxy:
            # Ultra-fast physics-based proxy (~100,000 eval/s)
            print(f"  Using FAST PHYSICS PROXY (no FAIRChem)")
            self.fast_proxy = FastPhysicsProxy(
                config=ProxyConfig(target_energy=target_energy, target_si_al_ratio=4.0),
                energy_model=self._aligned_energy_model,
            )
            self.proxy = None
            self.batched_evaluator = None
        elif use_gpu and torch.cuda.is_available():
            # Use GPU sequential evaluator (10-100x faster than CPU multiprocessing)
            print(f"  Using GPU sequential evaluation (CUDA)")
            self.proxy = FAIRChemProxy(
                checkpoint=oracle.checkpoint if hasattr(oracle, 'checkpoint') else "esen-sm-odac25-full",
                target_energy=oracle.target_energy if hasattr(oracle, 'target_energy') else -0.5,
                fmax=fmax,
                max_bfgs_steps=max_bfgs_steps,
                num_initial_positions=num_initial_positions,
                use_mock=use_mock_oracle,
                use_gpu=True,
            )
            self.batched_evaluator = None
        elif use_batching:
            # Use batched evaluator with multiprocessing
            # NOTE: Use CPU for workers to avoid CUDA multiprocessing issues with single GPU
            print(f"  Using CPU multiprocessing evaluation")
            self.proxy = None
            self.batched_evaluator = BatchedEvaluator(
                oracle_checkpoint=oracle.checkpoint if hasattr(oracle, 'checkpoint') else "esen-sm-odac25-full",
                target_energy=oracle.target_energy,
                fmax=fmax,
                max_bfgs_steps=max_bfgs_steps,
                num_initial_positions=num_initial_positions,
                num_workers=num_workers,
                use_mock=use_mock_oracle,
                worker_device="cpu"  # Use CPU for multiprocessing workers
            )
        elif use_bfgs:
            # Use single-threaded BFGS evaluator
            self.proxy = None
            self.bfgs_evaluator = BFGSRelaxedEvaluator(
                oracle=oracle,
                fmax=fmax,
                max_steps=max_bfgs_steps,
                num_initial_positions=num_initial_positions
            )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size, alpha=replay_alpha)
        self.oversample_factor = oversample_factor

        # Hybrid pre-filtering: fast proxy ranks candidates, real evaluator scores top ones.
        # Works with any evaluator (FAIRChem+BFGS, GPU, mock, etc.)
        # When use_fast_proxy=True (pure fast proxy mode), oversampling still works but
        # pre-filters with the same fast proxy before the main evaluation path.
        self.hybrid_mode = oversample_factor > 1
        self.hybrid_proxy = None
        if self.hybrid_mode:
            try:
                # Reuse the fast proxy if already created, otherwise create one for filtering
                if use_fast_proxy and hasattr(self, 'fast_proxy'):
                    self.hybrid_proxy = self.fast_proxy
                else:
                    self.hybrid_proxy = FastPhysicsProxy(
                        config=ProxyConfig(target_energy=target_energy, target_si_al_ratio=4.0),
                        energy_model=self._aligned_energy_model,
                    )
                evaluator_name = "fast proxy" if use_fast_proxy else (
                    "mock" if use_mock_oracle else "FAIRChem+BFGS")
                print(f"  Hybrid mode: oversample {oversample_factor}x with fast proxy → evaluate top with {evaluator_name}")
            except Exception as e:
                print(f"  Warning: Could not initialize hybrid proxy: {e}")
                self.hybrid_mode = False

        # Training statistics
        self.stats = {
            'iteration': [],
            'loss': [],
            'log_Z': [],
            'mean_reward': [],
            'max_reward': [],
            'mean_energy': [],
            'best_energy': [],
            'diversity': [],
            'mean_si_al_ratio': [],
        }

        print(f"Trainer initialized:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Fast proxy: {use_fast_proxy}")
        print(f"  Mock oracle: {use_mock_oracle}")
        print(f"  BFGS relaxation: {use_bfgs}")
        print(f"  GPU evaluation: {use_gpu and torch.cuda.is_available()}")
        print(f"  Batched evaluation: {use_batching}")
        if not use_fast_proxy:
            print(f"  max_bfgs_steps: {max_bfgs_steps}")
            print(f"  num_initial_positions: {num_initial_positions}")

    def evaluate_structure(self, state: FaujasiteState) -> Dict:
        """
        Evaluate a single Faujasite structure using the oracle.

        With BFGS: Places CO2, relaxes with BFGS, computes adsorption energy
        Without BFGS: Uses simple position sampling (legacy behavior)

        Args:
            state: FaujasiteState to evaluate

        Returns:
            dict with energy, reward, and metadata
        """
        # Convert state to ASE structure
        framework = self.env.state_to_atoms(state)

        if self.use_mock_oracle:
            # Mock evaluation for testing
            si_al_ratio = self.env.get_si_al_ratio(state)
            # Mock: reward structures with Si/Al around 5-10
            target_ratio = 7.5
            mock_energy = -0.5 + 0.1 * abs(si_al_ratio - target_ratio)
            sigma = getattr(self, '_current_sigma', 0.5)
            mock_reward = self.oracle.compute_reward(mock_energy, sigma=sigma)

            return {
                'energy': mock_energy,
                'reward': mock_reward,
                'si_al_ratio': si_al_ratio,
                'num_al': int(np.sum(state.config)),
                'converged': True,
            }

        elif self.use_batching:
            # Use batched evaluator for single structure
            try:
                results = self.batched_evaluator.evaluate_batch(
                    [framework],
                    adsorbate_name="CO2"
                )
                result = results[0]

                # Add state-specific metadata
                result['si_al_ratio'] = self.env.get_si_al_ratio(state)
                result['num_al'] = int(np.sum(state.config))

                return result

            except Exception as e:
                print(f"Batched evaluation failed: {e}")
                return {
                    'energy': 10.0,
                    'reward': 0.0,
                    'si_al_ratio': self.env.get_si_al_ratio(state),
                    'num_al': int(np.sum(state.config)),
                    'converged': False,
                    'error': str(e)
                }

        elif self.use_bfgs and hasattr(self, 'bfgs_evaluator'):
            # Use single-threaded BFGS evaluator
            try:
                result = self.bfgs_evaluator.evaluate_structure(
                    framework,
                    adsorbate_name="CO2",
                    return_relaxed_structure=False
                )

                # Add state-specific metadata
                result['si_al_ratio'] = self.env.get_si_al_ratio(state)
                result['num_al'] = int(np.sum(state.config))

                return result

            except Exception as e:
                print(f"BFGS evaluation failed: {e}")
                return {
                    'energy': 10.0,
                    'reward': 0.0,
                    'si_al_ratio': self.env.get_si_al_ratio(state),
                    'num_al': int(np.sum(state.config)),
                    'converged': False,
                    'error': str(e)
                }

        else:
            # Legacy: Simple position sampling without BFGS
            try:
                from ase.build import molecule
                from fairchem_oracle import place_adsorbate

                co2 = molecule("CO2")
                co2.center(vacuum=5.0)
                co2.cell = framework.cell.copy()
                co2.pbc = True

                E_framework = self.oracle.predict_energy(framework)
                E_co2 = self.oracle.predict_energy(co2)
                E_reference = E_framework + E_co2

                best_E_system = float('inf')
                for dist in [2.5, 3.0, 3.5]:
                    try:
                        system = place_adsorbate(framework, co2, distance=dist)
                        E_sys = self.oracle.predict_energy(system)
                        if E_sys < best_E_system:
                            best_E_system = E_sys
                    except:
                        continue

                E_ads = best_E_system - E_reference
                sigma = getattr(self, '_current_sigma', 0.5)
                reward = self.oracle.compute_reward(E_ads, sigma=sigma)

                return {
                    'energy': E_ads,
                    'reward': reward,
                    'si_al_ratio': self.env.get_si_al_ratio(state),
                    'num_al': int(np.sum(state.config)),
                    'converged': True,
                }
            except Exception as e:
                print(f"Oracle evaluation failed: {e}")
                return {
                    'energy': 10.0,
                    'reward': 0.0,
                    'si_al_ratio': self.env.get_si_al_ratio(state),
                    'num_al': int(np.sum(state.config)),
                    'converged': False,
                }

    def sample_trajectories(
        self, batch_size: int, temperature: float = 1.0,
        epsilon: float = 0.0, noise_scale: float = 0.0
    ) -> List[Trajectory]:
        """
        Sample a batch of trajectories and evaluate them.

        With batching: Evaluates all structures in parallel
        Without batching: Evaluates sequentially
        Hybrid mode: Oversample with fast proxy, keep top batch_size for FAIRChem

        Args:
            batch_size: Number of trajectories to sample
            temperature: Sampling temperature
            epsilon: Exploration rate for epsilon-uniform blending
            noise_scale: Gaussian noise scale for logit perturbation

        Returns:
            List of trajectories with rewards
        """
        # Determine how many to sample (oversample for hybrid pre-filtering)
        n_sample = batch_size
        if self.hybrid_mode and self.oversample_factor > 1:
            n_sample = batch_size * self.oversample_factor

        # Sample all trajectories first
        print(f"  Sampling {n_sample} trajectories...", flush=True)
        sampled_data = []
        for i in range(n_sample):
            states, actions, log_probs, terminal_lp = self.gfn.sample_trajectory(
                temperature, epsilon=epsilon, noise_scale=noise_scale
            )
            sampled_data.append((states, actions, log_probs, terminal_lp))
        print(f"  ✓ Sampled {n_sample} structures", flush=True)

        # Hybrid pre-filtering: rank by fast proxy, keep top batch_size
        if self.hybrid_mode and self.oversample_factor > 1 and n_sample > batch_size:
            sigma = getattr(self, '_current_sigma', 0.5)
            proxy_rewards = []
            for states, _, _, _ in sampled_data:
                final_state = states[-1]
                result = self.hybrid_proxy.evaluate(final_state.config, sigma=sigma)
                proxy_rewards.append(result['reward'])

            # Keep top batch_size by proxy reward
            top_indices = np.argsort(proxy_rewards)[-batch_size:]
            sampled_data = [sampled_data[i] for i in sorted(top_indices)]
            print(f"  Hybrid filter: {n_sample} → {batch_size} (top proxy rewards)", flush=True)

        # Evaluate structures
        if self.use_fast_proxy:
            # Ultra-fast physics proxy (no FAIRChem)
            sigma = getattr(self, '_current_sigma', 0.5)
            eval_results = []
            for states, _, _, _ in sampled_data:
                final_state = states[-1]
                result = self.fast_proxy.evaluate(final_state.config, sigma=sigma)
                eval_results.append(result)

        elif self.proxy is not None:
            # GPU sequential evaluation (fastest with FAIRChem)
            print(f"  Evaluating {batch_size} structures on GPU...", flush=True)
            frameworks = [self.env.state_to_atoms(states[-1]) for states, _, _, _ in sampled_data]
            eval_results = self.proxy.evaluate_batch(frameworks, adsorbate_name="CO2")

            # Add state-specific metadata to results
            for (states, _, _, _), result in zip(sampled_data, eval_results):
                final_state = states[-1]
                result['si_al_ratio'] = self.env.get_si_al_ratio(final_state)
                result['num_al'] = int(np.sum(final_state.config))

        elif self.use_batching and self.batched_evaluator is not None:
            # Batched evaluation with multiprocessing
            print(f"  Evaluating {batch_size} structures in parallel...", flush=True)
            frameworks = [self.env.state_to_atoms(states[-1]) for states, _, _, _ in sampled_data]
            eval_results = self.batched_evaluator.evaluate_batch(frameworks, adsorbate_name="CO2")

            # Add state-specific metadata to results
            for (states, _, _, _), result in zip(sampled_data, eval_results):
                final_state = states[-1]
                result['si_al_ratio'] = self.env.get_si_al_ratio(final_state)
                result['num_al'] = int(np.sum(final_state.config))

        else:
            # Sequential evaluation
            print(f"  Evaluating {batch_size} structures sequentially:", flush=True)
            eval_results = []
            eval_start = time.time()
            for idx, (states, _, _, _) in enumerate(sampled_data):
                struct_start = time.time()
                final_state = states[-1]
                result = self.evaluate_structure(final_state)
                eval_results.append(result)
                elapsed = time.time() - struct_start
                print(f"    [{idx+1}/{batch_size}] E={result['energy']:6.3f} eV, R={result['reward']:.4f}, t={elapsed:.1f}s", flush=True)
            total_eval_time = time.time() - eval_start
            print(f"  ✓ Evaluation complete ({total_eval_time:.1f}s total, {total_eval_time/batch_size:.1f}s avg)", flush=True)

        # Create trajectory objects
        trajectories = []
        for (states, actions, log_probs, terminal_lp), eval_result in zip(sampled_data, eval_results):
            traj = Trajectory(
                states=states,
                actions=actions,
                rewards=[0.0] * len(actions),
                final_reward=eval_result['reward'],
                sum_log_prob=(sum(log_probs) + terminal_lp) if log_probs else terminal_lp,
                terminal_log_prob=terminal_lp,
            )
            traj.eval_result = eval_result
            trajectories.append(traj)

        return trajectories

    def train_iteration(
        self, iteration: int, batch_size: int, temperature: float = 1.0,
        sigma: float = 0.5, epsilon: float = 0.0, noise_scale: float = 0.0,
        replay_ratio: float = 0.25
    ) -> Dict:
        """
        Single training iteration.

        Args:
            iteration: Current iteration number
            batch_size: Number of trajectories per batch
            temperature: Sampling temperature
            sigma: Reward function width (smaller = sharper peak around target energy)
            epsilon: Exploration rate for epsilon-uniform blending
            noise_scale: Gaussian noise scale for logit perturbation
            replay_ratio: Fraction of batch to sample from replay buffer

        Returns:
            Info dict with metrics
        """
        self._current_sigma = sigma  # Store for use in evaluate methods
        iter_start = time.time()

        # Sample trajectories
        sample_start = time.time()
        trajectories = self.sample_trajectories(
            batch_size, temperature, epsilon=epsilon, noise_scale=noise_scale
        )
        sample_time = time.time() - sample_start

        # Add to replay buffer
        self.replay_buffer.add(trajectories)

        # Sample from replay buffer for off-policy training
        n_replay = int(batch_size * replay_ratio)
        replay_trajs = self.replay_buffer.sample(n_replay) if n_replay > 0 and len(self.replay_buffer) > n_replay else None

        # Re-evaluate replay rewards with current sigma to prevent stale rewards.
        # Without this, replay trajectories stored at broad sigma (e.g. 0.5) have
        # inflated rewards that corrupt training when sigma has shrunk (e.g. 0.1).
        if replay_trajs:
            target = self.oracle.target_energy if hasattr(self.oracle, 'target_energy') else -0.5
            for traj in replay_trajs:
                if hasattr(traj, 'eval_result') and 'energy' in traj.eval_result:
                    energy = traj.eval_result['energy']
                    traj.final_reward = float(np.exp(-0.5 * ((energy - target) / sigma) ** 2))
                    traj.eval_result['reward'] = traj.final_reward

        # Train GFlowNet
        print(f"  Training GFlowNet...", flush=True)
        train_start = time.time()
        train_info = self.gfn.train_step(trajectories, replay_trajectories=replay_trajs)
        train_time = time.time() - train_start
        n_replay_used = len(replay_trajs) if replay_trajs else 0
        print(f"  ✓ Training step complete ({train_time:.1f}s, {len(trajectories)} fresh + {n_replay_used} replay)", flush=True)

        # Collect statistics
        energies = [t.eval_result['energy'] for t in trajectories]
        rewards = [t.eval_result['reward'] for t in trajectories]
        si_al_ratios = [t.eval_result['si_al_ratio'] for t in trajectories]

        # Diversity: unique configurations
        configs = [tuple(t.states[-1].config) for t in trajectories]
        diversity = len(set(configs)) / len(configs)

        iter_time = time.time() - iter_start

        info = {
            'iteration': iteration,
            'loss': train_info['loss'],
            'log_Z': train_info['log_Z'],
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'mean_energy': np.mean(energies),
            'best_energy': np.min(energies),  # Most negative
            'diversity': diversity,
            'mean_si_al_ratio': np.mean([r for r in si_al_ratios if not np.isinf(r)]),
            'sample_time': sample_time,
            'train_time': train_time,
            'iter_time': iter_time,
        }

        return info

    def train(
        self,
        num_iterations: int = 1000,
        batch_size: int = 16,
        log_interval: int = 10,
        save_interval: int = 100,
        temperature_schedule: str = 'constant',
        sigma_start: float = 0.5,
        sigma_end: float = 0.2,
        sigma_schedule: str = 'linear',
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        noise_scale: float = 0.0,
    ):
        """
        Main training loop.

        Args:
            num_iterations: Number of training iterations
            batch_size: Trajectories per iteration
            log_interval: Print stats every N iterations
            save_interval: Save checkpoint every N iterations
            temperature_schedule: 'constant', 'linear', or 'exponential'
            sigma_start: Initial reward sigma (broad peak for exploration)
            sigma_end: Final reward sigma (sharp peak for exploitation)
            sigma_schedule: 'constant', 'linear', or 'exponential'
            epsilon_start: Initial epsilon for uniform exploration
            epsilon_end: Final epsilon for uniform exploration
            noise_scale: Gaussian noise scale for logit perturbation
        """
        print("\n" + "="*70)
        print("Starting GFlowNet Training")
        print("="*70)
        print(f"  Iterations: {num_iterations}")
        print(f"  Batch size: {batch_size}")
        print(f"  Temperature schedule: {temperature_schedule}")
        print(f"  Sigma annealing: {sigma_start:.2f} → {sigma_end:.2f} ({sigma_schedule})")
        print(f"  Epsilon annealing: {epsilon_start:.2f} → {epsilon_end:.2f} (linear)")
        print(f"  Noise scale: {noise_scale:.2f}")
        print(f"  Replay buffer: {self.replay_buffer.max_size}")
        print()

        start_time = time.time()

        for iteration in range(1, num_iterations + 1):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}/{num_iterations}")
            print(f"{'='*70}")

            # Temperature schedule
            if temperature_schedule == 'linear':
                temperature = 1.0 - 0.5 * (iteration / num_iterations)
            elif temperature_schedule == 'exponential':
                temperature = np.exp(-2 * iteration / num_iterations)
            else:
                temperature = 1.0

            # Sigma annealing schedule
            progress = iteration / num_iterations
            if sigma_schedule == 'linear':
                sigma = sigma_start - (sigma_start - sigma_end) * progress
            elif sigma_schedule == 'exponential':
                sigma = sigma_end + (sigma_start - sigma_end) * np.exp(-3 * progress)
            else:
                sigma = sigma_start  # constant

            # Epsilon annealing (linear decay)
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress

            # Train iteration
            info = self.train_iteration(
                iteration, batch_size, temperature, sigma,
                epsilon=epsilon, noise_scale=noise_scale
            )

            # Update statistics
            for key, value in info.items():
                if key not in self.stats:
                    self.stats[key] = []
                self.stats[key].append(value)

            # Always log after each iteration
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"Iteration {iteration} Summary:")
            print(f"{'='*70}")
            print(f"  Loss:       {info['loss']:6.3f}")
            print(f"  Log Z:      {info['log_Z']:6.3f}")
            print(f"  Reward:     {info['mean_reward']:.4f} (max: {info['max_reward']:.4f})")
            print(f"  Energy:     {info['mean_energy']:6.3f} eV (best: {info['best_energy']:6.3f} eV)")
            print(f"  Diversity:  {info['diversity']:.2%}")
            print(f"  Si/Al:      {info['mean_si_al_ratio']:.2f}")
            print(f"  Sigma:      {sigma:.3f}")
            print(f"  Epsilon:    {epsilon:.3f}")
            print(f"  Replay buf: {len(self.replay_buffer)}")
            print(f"  Time:       Sample={info['sample_time']:.1f}s, Train={info['train_time']:.1f}s, Total={info['iter_time']:.1f}s")
            print(f"  Elapsed:    {elapsed:.1f}s ({elapsed/60:.1f}min)")

            # Estimate remaining time
            if iteration > 0:
                avg_iter_time = elapsed / iteration
                remaining = avg_iter_time * (num_iterations - iteration)
                print(f"  ETA:        {remaining:.0f}s ({remaining/60:.1f}min, {remaining/3600:.1f}hr)")
            print(f"{'='*70}")

            # Save checkpoint
            if iteration % save_interval == 0:
                self.save_checkpoint(iteration)

        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)

        # Final save
        self.save_checkpoint(num_iterations)
        self.save_results()

    def save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"gfn_checkpoint_{iteration}.pt"
        self.gfn.save_checkpoint(str(checkpoint_path))

    def save_results(self):
        """Save training results and statistics"""
        results_path = self.output_dir / "training_stats.json"

        with open(results_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"\nResults saved to {results_path}")

    def generate_structures(self, num_structures: int = 100, temperature: float = 0.8) -> List[Dict]:
        """
        Generate novel structures using trained GFlowNet.

        Args:
            num_structures: Number of structures to generate
            temperature: Sampling temperature (lower = more exploitation)

        Returns:
            List of structure dicts with energies and metadata
        """
        print(f"\nGenerating {num_structures} structures...")

        structures = []

        for i in range(num_structures):
            states, actions, _, _ = self.gfn.sample_trajectory(temperature)
            final_state = states[-1]

            eval_result = self.evaluate_structure(final_state)

            structure_data = {
                'index': i,
                'config': final_state.config.tolist(),
                'num_substitutions': final_state.num_substitutions,
                'si_al_ratio': eval_result['si_al_ratio'],
                'energy': eval_result['energy'],
                'reward': eval_result['reward'],
            }

            structures.append(structure_data)

            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{num_structures}")

        # Save structures
        structures_path = self.output_dir / "generated_structures.json"
        with open(structures_path, 'w') as f:
            json.dump(structures, f, indent=2)

        print(f"✓ Structures saved to {structures_path}")

        # Print summary
        energies = [s['energy'] for s in structures]
        print(f"\nGeneration Summary:")
        print(f"  Mean energy: {np.mean(energies):.3f} eV")
        print(f"  Best energy: {np.min(energies):.3f} eV")
        print(f"  Structures within ±0.1 eV of -0.5: {sum(1 for e in energies if abs(e + 0.5) < 0.1)}")

        return structures


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet for Faujasite discovery")
    parser.add_argument('--num_iterations', type=int, default=500, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (trajectories per iteration)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for GFlowNet')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_substitutions', type=int, default=5, help='Max Si→Al substitutions')
    parser.add_argument('--output_dir', type=str, default='gfn_results', help='Output directory')
    parser.add_argument('--mock_oracle', action='store_true', help='Use mock oracle (for testing)')
    parser.add_argument('--temperature_schedule', type=str, default='constant',
                        choices=['constant', 'linear', 'exponential'], help='Temperature schedule')
    parser.add_argument('--generate', type=int, default=0, help='Generate N structures after training')

    # Sigma annealing options
    parser.add_argument('--sigma_start', type=float, default=0.3,
                        help='Initial reward sigma (broader, more exploration)')
    parser.add_argument('--sigma_end', type=float, default=0.2,
                        help='Final reward sigma (sharper, more exploitation). '
                             'Too small kills gradient signal if energies are far from target.')
    parser.add_argument('--sigma_schedule', type=str, default='constant',
                        choices=['constant', 'linear', 'exponential'],
                        help='Sigma annealing schedule')

    # Exploration options
    parser.add_argument('--epsilon_start', type=float, default=0.3,
                        help='Initial epsilon for uniform exploration (default: 0.3)')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Final epsilon for uniform exploration (default: 0.01)')
    parser.add_argument('--noise_scale', type=float, default=0.0,
                        help='Gaussian noise scale for logit perturbation (default: 0.0, try 0.1-0.5)')

    # Replay buffer options
    parser.add_argument('--replay_buffer_size', type=int, default=1000,
                        help='Maximum replay buffer size (default: 1000)')
    parser.add_argument('--replay_alpha', type=float, default=1.0,
                        help='Replay priority exponent (default: 1.0)')

    # Hybrid pre-filtering: fast proxy selects candidates, real evaluator scores them
    parser.add_argument('--oversample_factor', type=int, default=1,
                        help='Oversample N*factor trajectories, pre-filter with fast proxy, '
                             'evaluate top N with real evaluator (FAIRChem+BFGS/mock). '
                             'Default: 1 (disabled), try 2-4 for hybrid mode')

    # BFGS and batching options
    parser.add_argument('--no_bfgs', action='store_true', help='Disable BFGS relaxation')
    parser.add_argument('--no_batching', action='store_true', help='Disable batched evaluation')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU evaluation (use CPU multiprocessing)')
    parser.add_argument('--fast_proxy', action='store_true',
                        help='Use fast physics proxy instead of FAIRChem (~1000x faster)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--fmax', type=float, default=0.05, help='BFGS force convergence (eV/Å)')
    parser.add_argument('--max_bfgs_steps', type=int, default=20, help='Max BFGS optimization steps (default: 20)')
    parser.add_argument('--num_initial_positions', type=int, default=1, help='Number of initial CO2 positions (default: 1)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("GFlowNet for Faujasite Discovery")
    print("="*70)

    # Initialize environment
    print("\n1. Initializing environment...")
    env = FaujasiteEnvironment(max_substitutions=args.max_substitutions)

    # Initialize GFlowNet
    print("\n2. Initializing GFlowNet...")
    gfn = GFlowNet(
        env=env,
        hidden_dim=args.hidden_dim,
        num_layers=3,
        learning_rate=args.learning_rate
    )

    # Initialize oracle
    print("\n3. Initializing oracle...")
    if args.fast_proxy:
        print("   Using FAST PHYSICS PROXY (no FAIRChem needed)")
        print("   ~100,000 evaluations/second - ideal for rapid exploration")
        # Create dummy oracle object for fast proxy mode
        oracle = type('DummyOracle', (), {
            'checkpoint': 'fast_proxy',
            'target_energy': -0.5,
            'device': 'cpu'
        })()
    elif args.mock_oracle:
        print("   Using MOCK oracle (for testing)")
        # Create dummy oracle object for mock mode
        oracle = type('MockOracle', (), {
            'checkpoint': 'mock',
            'target_energy': -0.5,
            'device': 'cpu'
        })()
    else:
        print("   Loading FAIRChem ODAC25 model...")
        print("   Note: If loading fails, the model cache may have an incompatible format")
        print("   Workaround: Delete .fairchem_cache and let it re-download, or use --mock_oracle")

        try:
            # Load ODAC25 model from HuggingFace (requires authentication)
            print("   Loading ODAC25 model from HuggingFace...")
            print("   Using: esen-sm-odac25-full")
            oracle = FAIRChemOracle(
                checkpoint="esen-sm-odac25-full",
                target_energy=-0.5,
                use_huggingface=True  # Download from HF with your credentials
            )
        except Exception as e:
            print(f"\n   ✗ Failed to load ODAC25 model: {e}")
            print("\n   Options:")
            print("     1. Use mock oracle: python train_gflownet.py --mock_oracle")
            print("     2. Clear cache: rm -rf .fairchem_cache && retry")
            print("     3. Use different checkpoint name (if available)")
            raise

    # Initialize trainer
    print("\n4. Initializing trainer...")
    trainer = GFlowNetTrainer(
        env=env,
        gfn=gfn,
        oracle=oracle,
        output_dir=args.output_dir,
        use_mock_oracle=args.mock_oracle,
        use_fast_proxy=args.fast_proxy,
        use_bfgs=not args.no_bfgs,
        use_batching=not args.no_batching,
        use_gpu=not args.no_gpu,
        num_workers=args.num_workers,
        fmax=args.fmax,
        max_bfgs_steps=args.max_bfgs_steps,
        num_initial_positions=args.num_initial_positions,
        replay_buffer_size=args.replay_buffer_size,
        replay_alpha=args.replay_alpha,
        oversample_factor=args.oversample_factor,
    )

    # Train
    print("\n5. Starting training...")
    # Initialize log Z with fast proxy if available
    if args.fast_proxy:
        print("\n4b. Initializing log Z with fast proxy...")
        gfn.initialize_log_z(trainer.fast_proxy)
    elif trainer.hybrid_proxy is not None:
        print("\n4b. Initializing log Z with hybrid proxy...")
        gfn.initialize_log_z(trainer.hybrid_proxy)

    trainer.train(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        log_interval=10,
        save_interval=100,
        temperature_schedule=args.temperature_schedule,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        sigma_schedule=args.sigma_schedule,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        noise_scale=args.noise_scale,
    )

    # Generate structures
    if args.generate > 0:
        print(f"\n6. Generating {args.generate} structures...")
        trainer.generate_structures(num_structures=args.generate, temperature=0.8)

    print("\n" + "="*70)
    print("All done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
