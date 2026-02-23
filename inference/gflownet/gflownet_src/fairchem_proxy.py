"""
FAIRChem Proxy for GFlowNet Training

This module provides a unified interface for structure evaluation that
automatically selects the best evaluation strategy:
- GPU Sequential: Fast GPU-based evaluation (default when CUDA available)
- CPU Multiprocessing: Parallel CPU evaluation (fallback)

The proxy abstracts away the complexity of choosing evaluators and
provides a simple interface for GFlowNet training.
"""

import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class FAIRChemProxy:
    """
    Proxy class for FAIRChem-based structure evaluation.

    Automatically selects the optimal evaluation strategy:
    - use_gpu=True (default): Uses GPUSequentialEvaluator for fast GPU inference
    - use_gpu=False: Uses BatchedEvaluator with CPU multiprocessing

    GPU evaluation is typically 10-100x faster than CPU for neural network
    inference, making it the preferred choice when CUDA is available.
    """

    def __init__(
        self,
        checkpoint: str = "esen-sm-odac25-full",
        target_energy: float = -0.5,
        fmax: float = 0.05,
        max_bfgs_steps: int = 20,
        num_initial_positions: int = 1,
        num_workers: Optional[int] = None,
        use_mock: bool = False,
        use_gpu: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize FAIRChem proxy for structure evaluation.

        Args:
            checkpoint: FAIRChem checkpoint name or path
            target_energy: Target adsorption energy (eV)
            fmax: BFGS force convergence (eV/Å)
            max_bfgs_steps: Maximum BFGS steps (default: 20 for speed)
            num_initial_positions: CO2 positions to try (default: 1 for speed)
            num_workers: Number of CPU workers (only used when use_gpu=False)
            use_mock: Use mock oracle for testing
            use_gpu: If True, use GPU sequential evaluation; if False, use CPU multiprocessing
            device: Override device selection (auto-detected if None)
        """
        self.checkpoint = checkpoint
        self.target_energy = target_energy
        self.fmax = fmax
        self.max_bfgs_steps = max_bfgs_steps
        self.num_initial_positions = num_initial_positions
        self.num_workers = num_workers
        self.use_mock = use_mock
        self.use_gpu = use_gpu

        # Device selection
        if device is not None:
            self.device = device
        elif use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            if use_gpu and not torch.cuda.is_available():
                logger.warning(
                    "use_gpu=True but CUDA not available. Falling back to CPU."
                )
                self.use_gpu = False

        # Lazy initialization of evaluator
        self._evaluator = None

        logger.info(f"FAIRChemProxy initialized:")
        logger.info(f"  use_gpu: {self.use_gpu}")
        logger.info(f"  device: {self.device}")
        logger.info(f"  checkpoint: {checkpoint}")
        logger.info(f"  max_bfgs_steps: {max_bfgs_steps}")
        logger.info(f"  num_initial_positions: {num_initial_positions}")

    def _get_evaluator(self):
        """Lazy initialization of the appropriate evaluator."""
        if self._evaluator is not None:
            return self._evaluator

        # Import evaluators here to avoid circular imports
        from src.evaluation.batched_evaluator import (
            GPUSequentialEvaluator,
            BatchedEvaluator,
        )

        if self.use_gpu:
            logger.info("Using GPUSequentialEvaluator for GPU-based evaluation")
            self._evaluator = GPUSequentialEvaluator(
                oracle_checkpoint=self.checkpoint,
                target_energy=self.target_energy,
                fmax=self.fmax,
                max_bfgs_steps=self.max_bfgs_steps,
                num_initial_positions=self.num_initial_positions,
                use_mock=self.use_mock,
                device=self.device,
            )
        else:
            logger.info("Using BatchedEvaluator for CPU multiprocessing evaluation")
            self._evaluator = BatchedEvaluator(
                oracle_checkpoint=self.checkpoint,
                target_energy=self.target_energy,
                fmax=self.fmax,
                max_bfgs_steps=self.max_bfgs_steps,
                num_initial_positions=self.num_initial_positions,
                num_workers=self.num_workers,
                use_mock=self.use_mock,
                worker_device="cpu",  # CPU workers for multiprocessing
            )

        return self._evaluator

    def evaluate_batch(
        self,
        frameworks: List,
        adsorbate_name: str = "CO2"
    ) -> List[Dict]:
        """
        Evaluate a batch of framework structures.

        Args:
            frameworks: List of ASE Atoms framework structures
            adsorbate_name: Adsorbate molecule name

        Returns:
            List of dicts with 'energy', 'reward', 'converged', etc.
        """
        evaluator = self._get_evaluator()
        return evaluator.evaluate_batch(frameworks, adsorbate_name=adsorbate_name)

    def evaluate_single(
        self,
        framework,
        adsorbate_name: str = "CO2"
    ) -> Dict:
        """
        Evaluate a single framework structure.

        Args:
            framework: ASE Atoms framework structure
            adsorbate_name: Adsorbate molecule name

        Returns:
            Dict with 'energy', 'reward', 'converged', etc.
        """
        evaluator = self._get_evaluator()

        # GPUSequentialEvaluator has evaluate_single, BatchedEvaluator doesn't
        if hasattr(evaluator, 'evaluate_single'):
            return evaluator.evaluate_single(framework, adsorbate_name=adsorbate_name)
        else:
            # Fall back to batch evaluation with single item
            results = evaluator.evaluate_batch([framework], adsorbate_name=adsorbate_name)
            return results[0] if results else {'energy': 10.0, 'reward': 0.0, 'converged': False}

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU evaluation."""
        return self.use_gpu and self.device == "cuda"

    def __repr__(self) -> str:
        mode = "GPU" if self.is_gpu else "CPU"
        return (
            f"FAIRChemProxy(mode={mode}, device={self.device}, "
            f"checkpoint={self.checkpoint})"
        )


def create_proxy_from_config(config: Dict) -> FAIRChemProxy:
    """
    Create FAIRChemProxy from a configuration dictionary.

    Args:
        config: Dictionary with proxy configuration

    Returns:
        Configured FAIRChemProxy instance
    """
    return FAIRChemProxy(
        checkpoint=config.get("checkpoint", "esen-sm-odac25-full"),
        target_energy=config.get("target_energy", -0.5),
        fmax=config.get("fmax", 0.05),
        max_bfgs_steps=config.get("max_bfgs_steps", 20),
        num_initial_positions=config.get("num_initial_positions", 1),
        num_workers=config.get("num_workers"),
        use_mock=config.get("use_mock", False),
        use_gpu=config.get("use_gpu", True),
        device=config.get("device"),
    )
