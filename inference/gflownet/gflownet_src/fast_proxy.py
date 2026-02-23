"""
Fast Physics-Based Proxy for GFlowNet Training (Modular Design)

This proxy provides instant reward estimation without FAIRChem,
enabling ~1000x faster training iterations.

Design principles:
- Modular: Swap reward components easily via configuration
- Extensible: Add new heuristics by subclassing
- Testable: Each component can be tested independently
- Deterministic: No random noise for consistent gradients

Usage:
    # Default proxy
    proxy = FastPhysicsProxy()

    # Custom configuration
    proxy = FastPhysicsProxy.from_config({
        'energy_model': 'quadratic',
        'use_dispersion': True,
        'use_supercage': True,
        'use_lowenstein': True,
    })

    # Or swap components directly
    proxy = FastPhysicsProxy(
        energy_model=QuadraticEnergyModel(optimal_al_frac=0.15),
        dispersion_scorer=SpatialDispersionScorer(),
    )
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProxyConfig:
    """Configuration for FastPhysicsProxy."""

    # Energy model selection
    energy_model: str = 'quadratic'  # 'linear', 'quadratic', 'piecewise'

    # Component weights (set to 0 to disable)
    dispersion_weight: float = 0.4
    supercage_weight: float = 0.3
    lowenstein_penalty: float = 1.0
    ratio_penalty_weight: float = 0.2

    # Target parameters
    target_energy: float = -0.5
    target_si_al_ratio: float = 4.0
    si_al_ratio_width: float = 3.0

    # Reward function
    reward_type: str = 'gaussian'  # 'gaussian', 'exponential', 'linear'
    reward_sigma: float = 0.3

    # Structure
    template_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'energy_model': self.energy_model,
            'dispersion_weight': self.dispersion_weight,
            'supercage_weight': self.supercage_weight,
            'lowenstein_penalty': self.lowenstein_penalty,
            'ratio_penalty_weight': self.ratio_penalty_weight,
            'target_energy': self.target_energy,
            'target_si_al_ratio': self.target_si_al_ratio,
            'si_al_ratio_width': self.si_al_ratio_width,
            'reward_type': self.reward_type,
            'reward_sigma': self.reward_sigma,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ProxyConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================

class EnergyModel(ABC):
    """Base class for energy estimation models."""

    @abstractmethod
    def estimate(self, al_fraction: float, **kwargs) -> float:
        """Estimate base energy from Al fraction."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging."""
        pass


class DispersionScorer(ABC):
    """Base class for Al dispersion scoring."""

    @abstractmethod
    def score(self, config: np.ndarray, positions: Optional[np.ndarray] = None) -> float:
        """Score dispersion of Al atoms. Returns [0, 1]."""
        pass


class CageClassifier(ABC):
    """Base class for cage-type classification."""

    @abstractmethod
    def score(self, config: np.ndarray) -> float:
        """Score based on Al placement in accessible cages. Returns [0, 1]."""
        pass


class RewardFunction(ABC):
    """Base class for reward computation."""

    @abstractmethod
    def compute(self, energy: float, target: float) -> float:
        """Compute reward from energy. Returns [0, 1]."""
        pass


# =============================================================================
# ENERGY MODELS
# =============================================================================

class LinearEnergyModel(EnergyModel):
    """Simple linear relationship: more Al = more negative energy."""

    def __init__(self, slope: float = -2.0, intercept: float = 0.5):
        self.slope = slope
        self.intercept = intercept

    @property
    def name(self) -> str:
        return 'linear'

    def estimate(self, al_fraction: float, **kwargs) -> float:
        return self.intercept + self.slope * al_fraction


class QuadraticEnergyModel(EnergyModel):
    """
    Quadratic model with optimum Al fraction.

    Physics: Too little Al = weak binding, too much = framework destabilization.
    E = a + b*x + c*x^2 with minimum at x = -b/(2c)
    """

    def __init__(
        self,
        optimal_al_frac: float = 0.15,
        min_energy: float = -0.25,
        zero_al_energy: float = 0.5,
    ):
        """
        Args:
            optimal_al_frac: Al fraction with lowest energy
            min_energy: Energy at optimal Al fraction
            zero_al_energy: Energy with no Al
        """
        self.optimal_al_frac = optimal_al_frac
        self.min_energy = min_energy

        # Solve for coefficients: E(0) = zero_al_energy, E(optimal) = min_energy
        # E = a + b*x + c*x^2
        # a = zero_al_energy
        # -b/(2c) = optimal_al_frac  =>  b = -2c * optimal_al_frac
        # a + b*opt + c*opt^2 = min_energy
        # => a - 2c*opt^2 + c*opt^2 = min_energy
        # => a - c*opt^2 = min_energy
        # => c = (a - min_energy) / opt^2
        self.a = zero_al_energy
        self.c = (zero_al_energy - min_energy) / (optimal_al_frac ** 2)
        self.b = -2 * self.c * optimal_al_frac

    @property
    def name(self) -> str:
        return 'quadratic'

    def estimate(self, al_fraction: float, **kwargs) -> float:
        return self.a + self.b * al_fraction + self.c * al_fraction ** 2


class PiecewiseEnergyModel(EnergyModel):
    """
    Piecewise linear model with distinct regimes.

    Regime 1 (0-10% Al): Rapid improvement
    Regime 2 (10-25% Al): Optimal range
    Regime 3 (>25% Al): Degradation
    """

    def __init__(self):
        # Breakpoints and slopes
        self.breakpoints = [0.0, 0.10, 0.25, 1.0]
        self.values = [0.3, -0.15, -0.20, 0.5]  # Energy at each breakpoint

    @property
    def name(self) -> str:
        return 'piecewise'

    def estimate(self, al_fraction: float, **kwargs) -> float:
        al_fraction = np.clip(al_fraction, 0.0, 1.0)

        for i in range(len(self.breakpoints) - 1):
            if al_fraction <= self.breakpoints[i + 1]:
                x0, x1 = self.breakpoints[i], self.breakpoints[i + 1]
                y0, y1 = self.values[i], self.values[i + 1]
                t = (al_fraction - x0) / (x1 - x0) if x1 > x0 else 0
                return y0 + t * (y1 - y0)

        return self.values[-1]


# =============================================================================
# DISPERSION SCORERS
# =============================================================================

class IndexDispersionScorer(DispersionScorer):
    """Simple dispersion based on array index spacing (fallback)."""

    def score(self, config: np.ndarray, positions: Optional[np.ndarray] = None) -> float:
        al_indices = np.where(config == 1)[0]
        n_al = len(al_indices)

        if n_al <= 1:
            return 1.0

        n_sites = len(config)
        al_indices_sorted = np.sort(al_indices)
        spacings = np.diff(al_indices_sorted)
        spacings = np.append(spacings, n_sites - al_indices_sorted[-1] + al_indices_sorted[0])

        if np.mean(spacings) > 0:
            cv = np.std(spacings) / np.mean(spacings)
            return float(np.exp(-cv))
        return 0.0


class SpatialDispersionScorer(DispersionScorer):
    """Dispersion based on real 3D spatial distances."""

    def __init__(self, min_dist_target: float = 5.0, mean_dist_target: float = 8.0):
        """
        Args:
            min_dist_target: Target minimum Al-Al distance (Å) for good score
            mean_dist_target: Target mean Al-Al distance (Å)
        """
        self.min_dist_target = min_dist_target
        self.mean_dist_target = mean_dist_target
        self._fallback = IndexDispersionScorer()

    def score(self, config: np.ndarray, positions: Optional[np.ndarray] = None) -> float:
        al_indices = np.where(config == 1)[0]
        n_al = len(al_indices)

        if n_al <= 1:
            return 1.0

        if positions is None:
            return self._fallback.score(config)

        # Get valid Al positions
        valid_indices = al_indices[al_indices < len(positions)]
        if len(valid_indices) < 2:
            return 1.0

        al_positions = positions[valid_indices]

        try:
            from scipy.spatial.distance import pdist
            distances = pdist(al_positions)
        except ImportError:
            return self._fallback.score(config)

        if len(distances) == 0:
            return 1.0

        min_dist = np.min(distances)
        mean_dist = np.mean(distances)

        # Sigmoid scores
        min_dist_score = 1.0 / (1.0 + np.exp(-(min_dist - self.min_dist_target) / 2.0))
        mean_dist_score = 1.0 / (1.0 + np.exp(-(mean_dist - self.mean_dist_target) / 3.0))

        return float(0.7 * min_dist_score + 0.3 * mean_dist_score)


# =============================================================================
# CAGE CLASSIFIERS
# =============================================================================

class NullCageClassifier(CageClassifier):
    """No cage classification (neutral score)."""

    def score(self, config: np.ndarray) -> float:
        return 0.5


class FAUCageClassifier(CageClassifier):
    """
    Classify FAU T-sites by supercage accessibility.

    Supercage centers in FAU are at fractional coordinates like (0.25, 0.25, 0.25).
    T-sites closer to these centers are more accessible for CO₂ binding.
    """

    def __init__(self, positions: np.ndarray, cell_length: float):
        """
        Args:
            positions: T-site positions in Cartesian coordinates
            cell_length: Cubic cell parameter (Å)
        """
        self.positions = positions
        self.cell_length = cell_length
        self.supercage_scores = self._compute_supercage_scores()

    def _compute_supercage_scores(self) -> np.ndarray:
        """Pre-compute supercage accessibility score for each T-site."""
        if self.positions is None:
            return None

        frac_positions = self.positions / self.cell_length
        n_sites = len(frac_positions)
        scores = np.zeros(n_sites)

        # Supercage centers (fractional coordinates)
        supercage_centers = np.array([
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.75], [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25], [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25], [0.75, 0.25, 0.25],
        ])

        for i, frac in enumerate(frac_positions):
            frac = frac % 1.0
            dists = np.linalg.norm(frac - supercage_centers, axis=1)
            min_dist = np.min(dists)
            scores[i] = np.exp(-min_dist / 0.2)

        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def score(self, config: np.ndarray) -> float:
        if self.supercage_scores is None:
            return 0.5

        al_indices = np.where(config == 1)[0]
        n_al = len(al_indices)

        if n_al == 0:
            return 0.0

        valid_indices = al_indices[al_indices < len(self.supercage_scores)]
        if len(valid_indices) == 0:
            return 0.5

        return float(np.mean(self.supercage_scores[valid_indices]))


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class GaussianReward(RewardFunction):
    """Gaussian reward centered at target energy."""

    def __init__(self, sigma: float = 0.3):
        self.sigma = sigma

    def compute(self, energy: float, target: float) -> float:
        delta = energy - target
        return float(np.exp(-0.5 * (delta / self.sigma) ** 2))


class ExponentialReward(RewardFunction):
    """Exponential decay from target."""

    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def compute(self, energy: float, target: float) -> float:
        delta = abs(energy - target)
        return float(np.exp(-delta / self.scale))


class LinearReward(RewardFunction):
    """Linear reward (clamped)."""

    def __init__(self, max_delta: float = 1.0):
        self.max_delta = max_delta

    def compute(self, energy: float, target: float) -> float:
        delta = abs(energy - target)
        return float(max(0.0, 1.0 - delta / self.max_delta))


class ThresholdReward(RewardFunction):
    """Binary reward: 1 if within threshold, 0 otherwise."""

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def compute(self, energy: float, target: float) -> float:
        return 1.0 if abs(energy - target) < self.threshold else 0.0


# =============================================================================
# MAIN PROXY CLASS
# =============================================================================

class FastPhysicsProxy:
    """
    Fast physics-based proxy for structure evaluation.

    Modular design allows easy swapping of:
    - Energy models (linear, quadratic, piecewise)
    - Dispersion scorers (index-based, spatial)
    - Cage classifiers (none, FAU-specific)
    - Reward functions (gaussian, exponential, linear, threshold)
    """

    def __init__(
        self,
        config: Optional[ProxyConfig] = None,
        energy_model: Optional[EnergyModel] = None,
        dispersion_scorer: Optional[DispersionScorer] = None,
        cage_classifier: Optional[CageClassifier] = None,
        reward_function: Optional[RewardFunction] = None,
    ):
        """
        Initialize proxy with optional custom components.

        Args:
            config: ProxyConfig instance (uses defaults if None)
            energy_model: Custom energy model (overrides config)
            dispersion_scorer: Custom dispersion scorer (overrides config)
            cage_classifier: Custom cage classifier (overrides config)
            reward_function: Custom reward function (overrides config)
        """
        self.config = config or ProxyConfig()

        # Load template structure
        self.template = None
        self.t_site_positions = None
        self.t_site_indices = None
        self.adjacency_matrix = None
        self._load_template(self.config.template_path)

        # Initialize components
        self.energy_model = energy_model or self._create_energy_model()
        self.dispersion_scorer = dispersion_scorer or self._create_dispersion_scorer()
        self.cage_classifier = cage_classifier or self._create_cage_classifier()
        self.reward_function = reward_function or self._create_reward_function()

    @classmethod
    def from_config(cls, config_dict: Dict) -> 'FastPhysicsProxy':
        """Create proxy from configuration dictionary."""
        config = ProxyConfig.from_dict(config_dict)
        return cls(config=config)

    def _create_energy_model(self) -> EnergyModel:
        """Create energy model from config."""
        model_name = self.config.energy_model.lower()
        if model_name == 'linear':
            return LinearEnergyModel()
        elif model_name == 'quadratic':
            return QuadraticEnergyModel()
        elif model_name == 'piecewise':
            return PiecewiseEnergyModel()
        else:
            raise ValueError(f"Unknown energy model: {model_name}")

    def _create_dispersion_scorer(self) -> DispersionScorer:
        """Create dispersion scorer."""
        if self.config.dispersion_weight <= 0:
            return IndexDispersionScorer()  # Cheap fallback if disabled

        if self.t_site_positions is not None:
            return SpatialDispersionScorer()
        return IndexDispersionScorer()

    def _create_cage_classifier(self) -> CageClassifier:
        """Create cage classifier."""
        if self.config.supercage_weight <= 0:
            return NullCageClassifier()

        if self.t_site_positions is not None and self.template is not None:
            cell = self.template.get_cell()
            return FAUCageClassifier(self.t_site_positions, cell[0, 0])
        return NullCageClassifier()

    def _create_reward_function(self) -> RewardFunction:
        """Create reward function from config."""
        reward_type = self.config.reward_type.lower()
        if reward_type == 'gaussian':
            return GaussianReward(sigma=self.config.reward_sigma)
        elif reward_type == 'exponential':
            return ExponentialReward(scale=self.config.reward_sigma)
        elif reward_type == 'linear':
            return LinearReward(max_delta=self.config.reward_sigma * 3)
        elif reward_type == 'threshold':
            return ThresholdReward(threshold=self.config.reward_sigma)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    def _load_template(self, template_path: Optional[str] = None):
        """Load FAU template and extract T-site information."""
        try:
            from ase.io import read
            from ase.neighborlist import neighbor_list
        except ImportError:
            return

        paths_to_try = [
            template_path,
            'data/cif/FAU.cif',
            Path(__file__).parent.parent.parent / 'data' / 'cif' / 'FAU.cif',
        ]

        for path in paths_to_try:
            if path and Path(path).exists():
                try:
                    self.template = read(str(path))
                    break
                except Exception:
                    continue

        if self.template is None:
            return

        symbols = self.template.get_chemical_symbols()
        positions = self.template.get_positions()

        self.t_site_indices = [i for i, s in enumerate(symbols) if s == 'Si']
        self.t_site_positions = positions[self.t_site_indices]

        self._build_adjacency()

    def _build_adjacency(self):
        """Build T-site adjacency matrix based on shared oxygen bridges."""
        if self.template is None:
            return

        try:
            from ase.neighborlist import neighbor_list
        except ImportError:
            return

        n_sites = len(self.t_site_indices)
        self.adjacency_matrix = np.zeros((n_sites, n_sites), dtype=bool)

        i_atoms, j_atoms = neighbor_list('ij', self.template, cutoff=2.0)

        atom_to_tsite = {atom_idx: t_idx for t_idx, atom_idx in enumerate(self.t_site_indices)}
        o_indices = set(i for i, s in enumerate(self.template.get_chemical_symbols()) if s == 'O')

        o_to_tsites = {o: set() for o in o_indices}
        for i, j in zip(i_atoms, j_atoms):
            if i in o_indices and j in atom_to_tsite:
                o_to_tsites[i].add(atom_to_tsite[j])
            elif j in o_indices and i in atom_to_tsite:
                o_to_tsites[j].add(atom_to_tsite[i])

        for o_idx, tsites in o_to_tsites.items():
            tsites = list(tsites)
            for i in range(len(tsites)):
                for j in range(i + 1, len(tsites)):
                    self.adjacency_matrix[tsites[i], tsites[j]] = True
                    self.adjacency_matrix[tsites[j], tsites[i]] = True

    def check_lowenstein(self, config: np.ndarray) -> Tuple[bool, int]:
        """Check Löwenstein's rule. Returns (compliant, num_violations)."""
        if self.adjacency_matrix is None:
            return True, 0

        al_indices = np.where(config == 1)[0]
        max_idx = min(len(config), self.adjacency_matrix.shape[0])
        violations = 0

        for i in al_indices:
            if i >= max_idx:
                continue
            for j in al_indices:
                if j >= max_idx:
                    continue
                if i < j and self.adjacency_matrix[i, j]:
                    violations += 1

        return violations == 0, violations

    def compute_si_al_score(self, config: np.ndarray) -> float:
        """Score based on Si/Al ratio proximity to target. Returns [0, 1]."""
        n_al = np.sum(config)
        n_si = len(config) - n_al

        if n_al == 0:
            return 0.0

        si_al_ratio = n_si / n_al
        score = np.exp(-0.5 * ((si_al_ratio - self.config.target_si_al_ratio) / self.config.si_al_ratio_width) ** 2)

        if si_al_ratio < 2.0:
            score *= 0.5 * si_al_ratio
        elif si_al_ratio > 15.0:
            score *= np.exp(-(si_al_ratio - 15.0) / 10.0)

        return float(score)

    def estimate_energy(self, config: np.ndarray) -> float:
        """
        Estimate adsorption energy from configuration.

        Combines:
        - Base energy from energy model
        - Dispersion bonus
        - Cage accessibility bonus
        - Si/Al ratio penalty
        - Löwenstein violations penalty
        """
        n_al = np.sum(config)
        n_total = len(config)

        if n_al == 0:
            return 0.5  # No Al = weak binding

        al_fraction = n_al / n_total

        # Base energy from model
        base_energy = self.energy_model.estimate(al_fraction)

        # Dispersion bonus
        dispersion = self.dispersion_scorer.score(config, self.t_site_positions)
        dispersion_term = -self.config.dispersion_weight * dispersion

        # Supercage bonus
        supercage_score = self.cage_classifier.score(config)
        supercage_term = -self.config.supercage_weight * supercage_score

        # Si/Al ratio penalty
        si_al_score = self.compute_si_al_score(config)
        ratio_term = self.config.ratio_penalty_weight * (1.0 - si_al_score)

        # Löwenstein penalty
        _, violations = self.check_lowenstein(config)
        lowenstein_term = self.config.lowenstein_penalty * violations

        energy = base_energy + dispersion_term + supercage_term + ratio_term + lowenstein_term
        return float(np.clip(energy, -2.0, 5.0))

    def compute_reward(self, energy: float) -> float:
        """Compute reward from energy."""
        return self.reward_function.compute(energy, self.config.target_energy)

    def evaluate(self, config: np.ndarray, sigma: Optional[float] = None) -> Dict:
        """Full evaluation of a configuration."""
        energy = self.estimate_energy(config)

        # Allow override of sigma for compatibility
        if sigma is not None:
            reward = GaussianReward(sigma=sigma).compute(energy, self.config.target_energy)
        else:
            reward = self.compute_reward(energy)

        n_al = int(np.sum(config))
        n_si = len(config) - n_al
        si_al_ratio = n_si / n_al if n_al > 0 else float('inf')
        compliant, violations = self.check_lowenstein(config)

        return {
            'energy': energy,
            'reward': reward,
            'si_al_ratio': si_al_ratio,
            'num_al': n_al,
            'dispersion': self.dispersion_scorer.score(config, self.t_site_positions),
            'si_al_score': self.compute_si_al_score(config),
            'supercage_score': self.cage_classifier.score(config),
            'lowenstein_compliant': compliant,
            'lowenstein_violations': violations,
            'converged': True,
        }

    def evaluate_batch(
        self,
        configs: List[np.ndarray],
        adjacency: Optional[np.ndarray] = None,
        sigma: Optional[float] = None
    ) -> List[Dict]:
        """Evaluate multiple configurations."""
        return [self.evaluate(config, sigma) for config in configs]

    def get_component_info(self) -> Dict:
        """Return info about active components for logging."""
        return {
            'energy_model': self.energy_model.name,
            'dispersion_weight': self.config.dispersion_weight,
            'supercage_weight': self.config.supercage_weight,
            'lowenstein_penalty': self.config.lowenstein_penalty,
            'reward_type': self.config.reward_type,
            'reward_sigma': self.config.reward_sigma,
            'target_energy': self.config.target_energy,
            'has_template': self.template is not None,
            'n_t_sites': len(self.t_site_indices) if self.t_site_indices else 0,
        }


# =============================================================================
# TRAINER ADAPTER
# =============================================================================

class FastProxyTrainer:
    """Adapter to use FastPhysicsProxy with GFlowNetTrainer."""

    def __init__(
        self,
        target_energy: float = -0.5,
        target_si_al_ratio: float = 4.0,
        config: Optional[ProxyConfig] = None,
        **kwargs
    ):
        if config is None:
            config = ProxyConfig(
                target_energy=target_energy,
                target_si_al_ratio=target_si_al_ratio,
            )
        self.proxy = FastPhysicsProxy(config=config)
        self.target_energy = target_energy

    def evaluate_batch(self, frameworks, adsorbate_name: str = "CO2") -> List[Dict]:
        """Evaluate batch of ASE Atoms structures."""
        results = []
        for framework in frameworks:
            config = self._atoms_to_config(framework)
            result = self.proxy.evaluate(config)
            results.append(result)
        return results

    def _atoms_to_config(self, atoms) -> np.ndarray:
        """Extract Si/Al binary config from ASE Atoms."""
        symbols = atoms.get_chemical_symbols()
        return np.array([1 if s == 'Al' else 0 for s in symbols if s in ('Si', 'Al')])


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_preset_config(name: str) -> ProxyConfig:
    """Get a preset configuration by name."""
    presets = {
        'default': ProxyConfig(),

        'simple': ProxyConfig(
            energy_model='linear',
            dispersion_weight=0.0,
            supercage_weight=0.0,
            lowenstein_penalty=0.5,
        ),

        'full_physics': ProxyConfig(
            energy_model='quadratic',
            dispersion_weight=0.4,
            supercage_weight=0.3,
            lowenstein_penalty=1.0,
            ratio_penalty_weight=0.2,
        ),

        'strict_lowenstein': ProxyConfig(
            energy_model='quadratic',
            lowenstein_penalty=5.0,  # Heavy penalty
        ),

        'exploration': ProxyConfig(
            energy_model='quadratic',
            reward_type='gaussian',
            reward_sigma=0.5,  # Wide sigma for more gradient
        ),

        'exploitation': ProxyConfig(
            energy_model='quadratic',
            reward_type='gaussian',
            reward_sigma=0.1,  # Narrow sigma for precise targeting
        ),
    }

    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

    return presets[name]


# =============================================================================
# TESTING
# =============================================================================

def _create_lowenstein_compliant_config(proxy: FastPhysicsProxy, target_n_al: int) -> np.ndarray:
    """Create a Löwenstein-compliant configuration by greedy placement."""
    n_sites = len(proxy.t_site_positions) if proxy.t_site_positions is not None else 30
    config = np.zeros(n_sites, dtype=int)

    if proxy.adjacency_matrix is None:
        # No adjacency info, just space them out
        spacing = max(1, n_sites // target_n_al)
        for i in range(target_n_al):
            config[i * spacing % n_sites] = 1
        return config

    # Greedy: add Al to sites that don't violate Löwenstein
    al_placed = 0
    for site_idx in range(n_sites):
        if al_placed >= target_n_al:
            break

        # Check if this site has an Al neighbor
        has_al_neighbor = False
        for neighbor_idx in range(n_sites):
            if proxy.adjacency_matrix[site_idx, neighbor_idx] and config[neighbor_idx] == 1:
                has_al_neighbor = True
                break

        if not has_al_neighbor:
            config[site_idx] = 1
            al_placed += 1

    return config


def test_fast_proxy():
    """Test the modular fast proxy."""
    print("Testing Modular FastPhysicsProxy")
    print("=" * 80)

    # Initialize proxy
    proxy = FastPhysicsProxy()
    n_sites = len(proxy.t_site_positions) if proxy.t_site_positions is not None else 30
    print(f"Loaded FAU template: {n_sites} T-sites")
    print(f"Adjacency: avg {proxy.adjacency_matrix.sum(axis=1).mean():.1f} neighbors/site")

    # Test Löwenstein-compliant configurations at different Al levels
    print("\n--- Löwenstein-Compliant Configurations ---")
    print(f"{'Al%':<6} {'nAl':<5} {'Energy':>8} {'Reward':>8} {'Si/Al':>8} {'Disp':>6} {'SCage':>6} {'Löwen':>6}")
    print("-" * 70)

    for al_pct in [5, 10, 15, 20, 25]:
        target_n_al = int(n_sites * al_pct / 100)
        config = _create_lowenstein_compliant_config(proxy, target_n_al)
        result = proxy.evaluate(config)

        lowen = "✓" if result['lowenstein_compliant'] else f"✗{result['lowenstein_violations']}"
        print(f"{al_pct:>4}%  {result['num_al']:<5} {result['energy']:>8.3f} {result['reward']:>8.3f} "
              f"{result['si_al_ratio']:>8.2f} {result['dispersion']:>6.3f} "
              f"{result['supercage_score']:>6.3f} {lowen:>6}")

    # Test preset configurations
    print("\n--- Preset Configurations ---")
    for preset_name in ['default', 'simple', 'full_physics', 'exploration']:
        config_obj = get_preset_config(preset_name)
        test_proxy = FastPhysicsProxy(config=config_obj)

        # Use 15% Al config
        test_config = _create_lowenstein_compliant_config(test_proxy, int(n_sites * 0.15))
        result = test_proxy.evaluate(test_config)

        info = test_proxy.get_component_info()
        print(f"{preset_name:<15} model={info['energy_model']:<10} "
              f"energy={result['energy']:.3f} reward={result['reward']:.3f}")

    # Test reward functions on same energy
    print("\n--- Reward Function Comparison ---")
    test_config = _create_lowenstein_compliant_config(proxy, int(n_sites * 0.15))
    energy = proxy.estimate_energy(test_config)
    print(f"Energy: {energy:.3f}, Target: {proxy.config.target_energy}")

    for reward_cls in [GaussianReward, ExponentialReward, LinearReward, ThresholdReward]:
        reward_fn = reward_cls()
        r = reward_fn.compute(energy, proxy.config.target_energy)
        print(f"  {reward_cls.__name__:<20}: {r:.3f}")

    # Test with violations (to verify penalty works)
    print("\n--- Löwenstein Penalty Test ---")
    bad_config = np.zeros(n_sites, dtype=int)
    bad_config[:10] = 1  # Clustered Al likely creates violations
    result = proxy.evaluate(bad_config)
    print(f"Clustered config: {result['lowenstein_violations']} violations, energy={result['energy']:.3f}")

    # Speed test
    print("\n--- Speed Test ---")
    import time
    test_config = _create_lowenstein_compliant_config(proxy, int(n_sites * 0.15))
    n_evals = 10000
    start = time.time()
    for _ in range(n_evals):
        proxy.evaluate(test_config)
    elapsed = time.time() - start
    print(f"{n_evals} evaluations in {elapsed:.3f}s ({n_evals/elapsed:.0f} eval/s)")

    # Determinism check
    print("\n--- Determinism Check ---")
    e1 = proxy.estimate_energy(test_config)
    e2 = proxy.estimate_energy(test_config)
    print(f"Same config, two evals: {e1:.6f}, {e2:.6f}")
    print(f"Deterministic: {'✓' if e1 == e2 else '✗'}")

    print("\n" + "=" * 80)
    print("Tests complete!")


if __name__ == "__main__":
    test_fast_proxy()
