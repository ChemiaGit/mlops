#!/usr/bin/env python3
"""
Faujasite Structure Generation Environment for GFlowNet

This module defines the action space and state representation for generating
Faujasite zeolite structures through sequential Si/Al substitutions.

Key concepts:
- State: Current configuration of Si/Al at T-sites
- Action: Substitute Si → Al at a specific T-site
- Terminal: When Si/Al ratio constraint is met or max substitutions reached
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from pathlib import Path


@dataclass
class FaujasiteState:
    """Represents a Faujasite structure during generation"""

    # Si/Al configuration: 1 = Al, 0 = Si at each T-site
    config: np.ndarray  # Shape: (num_t_sites,)

    # Number of substitutions made so far
    num_substitutions: int

    # Track which sites have been modified
    modified_sites: List[int]

    def __hash__(self):
        return hash((tuple(self.config), self.num_substitutions))

    def clone(self):
        """Deep copy of state"""
        return FaujasiteState(
            config=self.config.copy(),
            num_substitutions=self.num_substitutions,
            modified_sites=self.modified_sites.copy()
        )


class FaujasiteEnvironment:
    """
    Environment for generating Faujasite structures via GFlowNet.

    Action space: Substitute Si → Al at each of the T-sites in the unit cell.

    Si/Al Ratio Bounds:
        - Theoretical minimum: Si/Al = 1.0 (50% Al, perfect alternation per Löwenstein)
        - Practical FAU ranges:
            * LSX zeolites: Si/Al ≈ 1.0–1.15
            * Zeolite X: Si/Al ≈ 1.2–1.5
            * Zeolite Y: Si/Al ≈ 1.5–3.0
            * USY (dealuminated): Si/Al > 30
        - Default range (1.0–30.0) covers adsorption-relevant compositions

    Note: Löwenstein's rule (no Al-O-Al) is the fundamental constraint, not Si/Al ratio.
    The ratio bounds define the search space for functionally relevant structures.
    """

    def __init__(
        self,
        template_path: Optional[str] = None,
        min_si_al_ratio: float = 1.0,
        max_si_al_ratio: float = 30.0,
        max_substitutions: int = 8,
        enforce_lowenstein: bool = True,
        use_supercage_sites: bool = True,
    ):
        """
        Initialize Faujasite environment.

        Args:
            template_path: Path to pristine FAU structure (all-Si)
            min_si_al_ratio: Minimum Si/Al ratio (default: 1.0, Löwenstein limit)
            max_si_al_ratio: Maximum Si/Al ratio (default: 30.0, adsorption-relevant)
            max_substitutions: Maximum number of Si→Al substitutions (default: 8, Si/Al 3-8 range)
            enforce_lowenstein: If True, enforce Löwenstein's rule (no Al-O-Al)
            use_supercage_sites: If True, filter T-sites by supercage proximity (keep top ~60%)
        """
        self.min_si_al_ratio = min_si_al_ratio
        self.max_si_al_ratio = max_si_al_ratio
        self.max_substitutions = max_substitutions
        self.enforce_lowenstein = enforce_lowenstein
        self.use_supercage_sites = use_supercage_sites

        # Load template structure
        if template_path and Path(template_path).exists():
            self.template = read(template_path)
        else:
            # Create a simplified Faujasite template
            self.template = self._create_simple_template()

        # Identify T-sites (tetrahedral Si/Al positions)
        self.t_sites = self._identify_t_sites()

        # Filter to supercage-accessible sites if requested
        if self.use_supercage_sites:
            self.t_sites = self._filter_supercage_sites(self.t_sites)

        self.num_t_sites = len(self.t_sites)

        # Build T-site adjacency for Löwenstein's rule
        if self.enforce_lowenstein:
            self.t_site_neighbors = self._build_t_site_adjacency()
        else:
            self.t_site_neighbors = {}

        print(f"Faujasite Environment initialized:")
        print(f"  T-sites: {self.num_t_sites}")
        print(f"  Si/Al ratio range: {self.min_si_al_ratio:.1f} - {self.max_si_al_ratio:.1f}")
        print(f"  Max substitutions: {self.max_substitutions}")
        print(f"  Supercage filtering: {'enabled' if self.use_supercage_sites else 'disabled'}")
        print(f"  Löwenstein's rule: {'enforced' if self.enforce_lowenstein else 'disabled'}")

    def _create_simple_template(self) -> Atoms:
        """
        Create a simplified Faujasite template.
        Try to load real FAU structure, fallback to simple structure.
        """
        # Try to load real Faujasite structure (prefer SiO2 version)
        # Look in multiple possible locations
        fau_paths = [
            'data/cif/FAU.cif',                        # Standard location
            Path(__file__).parent.parent.parent / 'data' / 'cif' / 'FAU.cif',  # Relative to module
            'fau_adsorption_prediction/FAU_SiO2.cif',  # Complete structure with O
            'FAU_SiO2.cif',
            'fau_adsorption_prediction/FAU.cif',       # Fallback: Si-only
            'FAU.cif',
        ]

        for path in fau_paths:
            if Path(path).exists():
                try:
                    template = read(path)
                    print(f"  Loaded real FAU structure from {path}")
                    print(f"  Atoms: {len(template)}, Formula: {template.get_chemical_formula()}")
                    return template
                except Exception as e:
                    print(f"  Warning: Could not load {path}: {e}")

        # Fallback: Create a simple cubic Si structure with proper spacing
        print("  Warning: Using simplified template (no real FAU found)")
        from ase.build import bulk

        # Small cubic structure with realistic Si-Si distances (~3.5 Å)
        template = bulk('Si', 'fcc', a=5.4, cubic=True)
        template = template * (2, 2, 2)  # Make it bigger

        return template

    def _identify_t_sites(self) -> List[int]:
        """
        Identify tetrahedral (T) sites in the structure.
        These are the Si/Al positions that can be substituted.
        """
        # In real Faujasite: T-sites are Si/Al positions in tetrahedral coordination
        # For simplicity, we'll treat all Si atoms as potential T-sites

        t_sites = []
        for i, atom in enumerate(self.template):
            if atom.symbol == 'Si':
                t_sites.append(i)

        # For real FAU (192 atoms), we'll use a subset for computational efficiency
        # Randomly sample T-sites to keep the action space manageable
        num_sites = min(len(t_sites), max(self.max_substitutions * 3, 30))

        if len(t_sites) > num_sites:
            import random
            random.seed(42)  # Reproducible sampling
            t_sites = random.sample(t_sites, num_sites)
            t_sites.sort()  # Keep them in order

        print(f"  Selected {len(t_sites)} T-sites from {len([a for a in self.template if a.symbol == 'Si'])} Si atoms")

        return t_sites

    def _filter_supercage_sites(self, t_sites: List[int]) -> List[int]:
        """
        Filter T-sites to keep only those near supercage centers.

        Supercage-proximal sites are more accessible for CO₂ binding and
        more relevant for adsorption optimization. Reduces action space
        by ~40%, significantly improving convergence speed.

        Uses the same supercage center logic as FAUCageClassifier in fast_proxy.py.

        Args:
            t_sites: Full list of T-site atom indices

        Returns:
            Filtered list of T-site indices (top ~60% by supercage proximity)
        """
        if len(t_sites) == 0:
            return t_sites

        positions = self.template.get_positions()
        cell = self.template.get_cell()

        # Get cell length (assume roughly cubic for FAU)
        cell_length = cell[0, 0] if cell[0, 0] > 0 else np.linalg.norm(cell[0])

        if cell_length <= 0:
            print("  Supercage filtering skipped: invalid cell")
            return t_sites

        # Supercage centers in fractional coordinates (FAU topology)
        supercage_centers = np.array([
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.75], [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25], [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25], [0.75, 0.25, 0.25],
        ])

        # Score each T-site by proximity to nearest supercage center
        scores = []
        for atom_idx in t_sites:
            frac = (positions[atom_idx] / cell_length) % 1.0
            dists = np.linalg.norm(frac - supercage_centers, axis=1)
            min_dist = np.min(dists)
            score = np.exp(-min_dist / 0.2)
            scores.append(score)

        scores = np.array(scores)

        # Keep top 60% of sites by score
        keep_fraction = 0.6
        n_keep = max(self.max_substitutions + 1, int(len(t_sites) * keep_fraction))
        n_keep = min(n_keep, len(t_sites))

        top_indices = np.argsort(scores)[-n_keep:]
        top_indices.sort()  # Keep original order

        filtered = [t_sites[i] for i in top_indices]

        print(f"  Supercage filter: {len(t_sites)} → {len(filtered)} T-sites (top {keep_fraction:.0%} by proximity)")

        return filtered

    def _build_t_site_adjacency(self) -> dict:
        """
        Build adjacency map for T-sites based on shared oxygen bridges.

        Two T-sites are adjacent if they share a bridging oxygen atom.
        This is used to enforce Löwenstein's rule (no Al-O-Al linkages).

        Returns:
            Dictionary mapping t_site_index -> set of neighboring t_site_indices
        """
        # Check if structure has oxygen atoms
        o_indices = [i for i, atom in enumerate(self.template) if atom.symbol == 'O']

        if not o_indices:
            # No oxygen atoms - use distance-based heuristic
            # T-T distance through bridging O is typically ~3.0-3.2 Å
            print("  No oxygen atoms found, using distance-based T-T adjacency")
            return self._build_distance_based_adjacency(cutoff=3.5)

        # Build neighbor list: find Si-O bonds (typical Si-O distance: 1.6-1.7 Å)
        # Use cutoff of 2.0 Å to be safe
        i_atoms, j_atoms = neighbor_list('ij', self.template, cutoff=2.0)

        # Map: atom_index -> set of bonded atom indices
        bonds = {}
        for i, j in zip(i_atoms, j_atoms):
            if i not in bonds:
                bonds[i] = set()
            bonds[i].add(j)

        # For each T-site, find which oxygens it's bonded to
        t_site_to_oxygens = {}
        for t_idx, atom_idx in enumerate(self.t_sites):
            bonded = bonds.get(atom_idx, set())
            # Keep only oxygen neighbors
            t_site_to_oxygens[t_idx] = {j for j in bonded if self.template[j].symbol == 'O'}

        # Two T-sites are adjacent if they share an oxygen
        t_site_neighbors = {t_idx: set() for t_idx in range(len(self.t_sites))}

        for t1 in range(len(self.t_sites)):
            for t2 in range(t1 + 1, len(self.t_sites)):
                # Check if they share an oxygen
                shared_oxygens = t_site_to_oxygens[t1] & t_site_to_oxygens[t2]
                if shared_oxygens:
                    t_site_neighbors[t1].add(t2)
                    t_site_neighbors[t2].add(t1)

        # Statistics
        total_edges = sum(len(neighbors) for neighbors in t_site_neighbors.values()) // 2
        avg_neighbors = np.mean([len(n) for n in t_site_neighbors.values()])
        print(f"  T-site connectivity: {total_edges} T-O-T bridges, avg {avg_neighbors:.1f} neighbors/site")

        return t_site_neighbors

    def _build_distance_based_adjacency(self, cutoff: float = 3.5) -> dict:
        """
        Fallback: Build T-site adjacency based on direct T-T distances.
        Used when structure has no oxygen atoms.

        Args:
            cutoff: Maximum T-T distance to consider as neighbors (Å)

        Returns:
            Dictionary mapping t_site_index -> set of neighboring t_site_indices
        """
        positions = self.template.get_positions()
        cell = self.template.get_cell()
        pbc = self.template.get_pbc()

        t_site_neighbors = {t_idx: set() for t_idx in range(len(self.t_sites))}

        for t1 in range(len(self.t_sites)):
            pos1 = positions[self.t_sites[t1]]
            for t2 in range(t1 + 1, len(self.t_sites)):
                pos2 = positions[self.t_sites[t2]]

                # Calculate minimum image distance (handles PBC)
                diff = pos2 - pos1
                if any(pbc):
                    # Apply minimum image convention
                    scaled = np.linalg.solve(cell.T, diff)
                    scaled -= np.round(scaled)
                    diff = cell.T @ scaled

                dist = np.linalg.norm(diff)

                if dist < cutoff:
                    t_site_neighbors[t1].add(t2)
                    t_site_neighbors[t2].add(t1)

        total_edges = sum(len(neighbors) for neighbors in t_site_neighbors.values()) // 2
        avg_neighbors = np.mean([len(n) for n in t_site_neighbors.values()])
        print(f"  T-site connectivity (distance-based): {total_edges} pairs, avg {avg_neighbors:.1f} neighbors/site")

        return t_site_neighbors

    def reset(self) -> FaujasiteState:
        """
        Reset to initial state (pristine all-Si structure).

        Returns:
            Initial FaujasiteState
        """
        return FaujasiteState(
            config=np.zeros(self.num_t_sites, dtype=np.int32),
            num_substitutions=0,
            modified_sites=[]
        )

    def get_valid_actions(self, state: FaujasiteState) -> List[int]:
        """
        Get list of valid actions from current state.

        Args:
            state: Current Faujasite state

        Returns:
            List of valid action indices (T-site indices)
        """
        if self.is_terminal(state):
            return []

        valid_actions = []

        for site_idx in range(self.num_t_sites):
            # Can only substitute Si → Al (not already Al)
            if state.config[site_idx] == 0:
                # Check if substitution would violate Si/Al ratio
                if self._is_valid_substitution(state, site_idx):
                    valid_actions.append(site_idx)

        # Always allow termination action
        valid_actions.append(self.num_t_sites)  # Terminal action

        return valid_actions

    def _is_valid_substitution(self, state: FaujasiteState, site_idx: int) -> bool:
        """
        Check if substituting this site is valid.

        Constraints checked:
        1. Si/Al ratio must stay within bounds (default: 1.0-30.0)
        2. Löwenstein's rule: no Al-O-Al linkages (if enforced)

        Note: Löwenstein's rule is the fundamental physical constraint.
        The Si/Al ratio bounds define the search space for relevant structures.
        """
        # Check Si/Al ratio constraint
        num_al = np.sum(state.config) + 1  # After this substitution
        num_si = self.num_t_sites - num_al

        if num_al > 0:
            si_al_ratio = num_si / num_al
            if not (self.min_si_al_ratio <= si_al_ratio <= self.max_si_al_ratio):
                return False

        # Check Löwenstein's rule: no Al-O-Al linkages
        if self.enforce_lowenstein:
            if not self._check_lowenstein(state, site_idx):
                return False

        return True

    def _check_lowenstein(self, state: FaujasiteState, site_idx: int) -> bool:
        """
        Check if placing Al at site_idx would violate Löwenstein's rule.

        Löwenstein's rule: Al-O-Al linkages are forbidden in zeolites.
        This means two adjacent T-sites (connected via bridging oxygen)
        cannot both be Al.

        This is the fundamental constraint that limits Al content. With perfect
        enforcement, the theoretical minimum Si/Al ratio is 1.0 (50% Al in
        perfect Si-Al alternation). In practice, FAU topology and synthesis
        chemistry typically yield Si/Al ≥ 1.0-1.2 for LSX/X zeolites.

        Args:
            state: Current state
            site_idx: T-site index where we want to place Al

        Returns:
            True if placement is allowed, False if it would create Al-O-Al
        """
        # Get neighboring T-sites
        neighbors = self.t_site_neighbors.get(site_idx, set())

        # Check if any neighbor is already Al
        for neighbor_idx in neighbors:
            if state.config[neighbor_idx] == 1:  # Neighbor is Al
                return False  # Would create Al-O-Al

        return True

    def step(self, state: FaujasiteState, action: int) -> FaujasiteState:
        """
        Apply action to state, returning new state.

        Args:
            state: Current state
            action: Action index (T-site to substitute, or terminal action)

        Returns:
            New state after action
        """
        new_state = state.clone()

        # Terminal action
        if action >= self.num_t_sites:
            # Mark as terminal (implementation detail)
            return new_state

        # Substitute Si → Al at this site
        if state.config[action] == 0:  # Only if currently Si
            new_state.config[action] = 1  # Change to Al
            new_state.num_substitutions += 1
            new_state.modified_sites.append(action)

        return new_state

    def is_terminal(self, state: FaujasiteState) -> bool:
        """
        Check if state is terminal.

        Terminal conditions:
        - Max substitutions reached
        - No more valid actions (Si/Al ratio constraints)
        """
        if state.num_substitutions >= self.max_substitutions:
            return True

        # Check if any valid substitutions remain
        num_al = np.sum(state.config)
        num_si = self.num_t_sites - num_al

        if num_al > 0:
            si_al_ratio = num_si / num_al
            # If ratio already at minimum, can't add more Al
            if si_al_ratio <= self.min_si_al_ratio:
                return True

        return False

    def state_to_atoms(self, state: FaujasiteState) -> Atoms:
        """
        Convert state to ASE Atoms object (actual structure).

        Args:
            state: FaujasiteState

        Returns:
            ASE Atoms object with Si/Al substitutions applied
        """
        structure = self.template.copy()

        # Apply Si → Al substitutions
        for site_idx, is_al in enumerate(state.config):
            if is_al == 1:
                atom_idx = self.t_sites[site_idx]
                structure[atom_idx].symbol = 'Al'

        return structure

    def get_si_al_ratio(self, state: FaujasiteState) -> float:
        """Calculate current Si/Al ratio"""
        num_al = np.sum(state.config)
        if num_al == 0:
            return float('inf')

        num_si = self.num_t_sites - num_al
        return num_si / num_al

    def state_to_tensor(self, state: FaujasiteState) -> np.ndarray:
        """
        Convert state to tensor representation for neural network.

        Returns:
            Feature vector: [config, num_substitutions, si_al_ratio]
        """
        si_al_ratio = self.get_si_al_ratio(state)
        if np.isinf(si_al_ratio):
            si_al_ratio = 100.0  # Cap for numerical stability

        features = np.concatenate([
            state.config.astype(np.float32),
            [state.num_substitutions / self.max_substitutions],
            [si_al_ratio / self.max_si_al_ratio]
        ])

        return features

    def get_state_dim(self) -> int:
        """Get dimensionality of state representation"""
        return self.num_t_sites + 2  # config + num_subs + si_al_ratio


def test_environment():
    """Test the Faujasite environment"""
    print("\n" + "="*70)
    print("Testing Faujasite Environment")
    print("="*70)

    env = FaujasiteEnvironment(max_substitutions=5, enforce_lowenstein=True)

    # Test reset
    state = env.reset()
    print(f"\nInitial state:")
    print(f"  Config: {state.config}")
    print(f"  Substitutions: {state.num_substitutions}")
    print(f"  Si/Al ratio: {env.get_si_al_ratio(state)}")

    # Test trajectory
    print(f"\nGenerating random trajectory (with Löwenstein's rule):")
    for step in range(5):
        valid_actions = env.get_valid_actions(state)
        print(f"  Step {step}: {len(valid_actions)} valid actions")

        if not valid_actions:
            break

        # Random action (exclude terminal if other actions available)
        non_terminal_actions = [a for a in valid_actions if a < env.num_t_sites]
        if not non_terminal_actions:
            print(f"    No more valid actions (constraints)")
            break

        action = np.random.choice(non_terminal_actions)
        state = env.step(state, action)

        print(f"    Action: {action} → Si/Al = {env.get_si_al_ratio(state):.2f}")

    # Test conversion to structure
    structure = env.state_to_atoms(state)
    print(f"\nFinal structure:")
    print(f"  Formula: {structure.get_chemical_formula()}")
    print(f"  Al atoms: {np.sum(state.config)}")

    # Test state representation
    tensor = env.state_to_tensor(state)
    print(f"\nState tensor shape: {tensor.shape}")
    print(f"State dimension: {env.get_state_dim()}")

    # Test Löwenstein validation
    print(f"\n--- Löwenstein's Rule Validation ---")
    _test_lowenstein_rule(env)

    print("\n" + "="*70)
    print("Environment test complete!")
    print("="*70 + "\n")


def _test_lowenstein_rule(env: FaujasiteEnvironment):
    """Test that Löwenstein's rule is properly enforced"""
    state = env.reset()

    # Place Al at first site
    if env.num_t_sites > 0:
        state = env.step(state, 0)
        print(f"  Placed Al at site 0")

        # Check which sites are now blocked due to Löwenstein
        blocked_sites = []
        for site_idx in range(env.num_t_sites):
            if state.config[site_idx] == 0:  # Si site
                if not env._check_lowenstein(state, site_idx):
                    blocked_sites.append(site_idx)

        neighbors_of_0 = env.t_site_neighbors.get(0, set())
        print(f"  Neighbors of site 0: {neighbors_of_0}")
        print(f"  Sites blocked by Löwenstein: {blocked_sites}")

        # Verify blocked sites match neighbors
        if set(blocked_sites) == neighbors_of_0:
            print(f"  ✓ Löwenstein's rule correctly blocks adjacent sites")
        else:
            print(f"  ✗ Mismatch: blocked={blocked_sites}, neighbors={neighbors_of_0}")

        # Verify blocked sites are NOT in valid actions
        valid_actions = env.get_valid_actions(state)
        valid_substitutions = [a for a in valid_actions if a < env.num_t_sites]
        blocked_in_valid = [s for s in blocked_sites if s in valid_substitutions]
        if not blocked_in_valid:
            print(f"  ✓ Blocked sites correctly excluded from valid actions")
        else:
            print(f"  ✗ Blocked sites wrongly in valid actions: {blocked_in_valid}")


if __name__ == "__main__":
    test_environment()
