"""
Azure ML scoring script for the GFlowNet zeolite generator.

The model generates or evaluates Faujasite (FAU) Si/Al configurations.
Source model: mat-gen/src/gflownet/

To deploy this endpoint, the mat-gen source must be bundled:
    cp -r /path/to/mat-gen/src/gflownet ./inference/gflownet/gflownet_src/

Then set code_configuration.code = "./inference/gflownet" in deploy_model.py.

Input JSON — generate new structures:
    {
        "mode": "generate",
        "n_samples": 5,
        "temperature": 1.0       # optional, default 1.0
    }

Input JSON — score a specific configuration:
    {
        "mode": "score",
        "config": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]   # binary Si/Al array
    }

Output JSON (generate):
    {
        "mode": "generate",
        "samples": [
            {
                "config": [0, 1, ...],
                "si_al_ratio": 5.0,
                "n_substitutions": 2,
                "reward": 0.83,
                "trajectory_length": 3
            }, ...
        ]
    }

Output JSON (score):
    {
        "mode": "score",
        "config": [0, 1, ...],
        "reward": 0.83,
        "si_al_ratio": 5.0
    }
"""

import io
import json
import logging
import os
import sys
import types

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Globals loaded in init()
gfn = None
env = None
fast_proxy = None
device = None

# Default generation parameters
DEFAULT_TEMPERATURE = 1.0
DEFAULT_N_SAMPLES = 1


# ---------------------------------------------------------------------------
# Azure ML entrypoints
# ---------------------------------------------------------------------------

def init():
    """Load GFlowNet model. Called once when endpoint starts."""
    global gfn, env, device, fast_proxy

    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, model dir: {model_dir}")

    # Add bundled source to sys.path
    src_dir = os.path.join(os.path.dirname(__file__), "gflownet_src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.info(f"Added {src_dir} to sys.path")

    # model.py internally imports "from src.gflownet.environment import ..."
    # (training repo structure: src/gflownet/).  Create a sys.modules alias so
    # those imports resolve to our bundled gflownet_src/ files.
    _src_pkg = types.ModuleType("src")
    _src_pkg.__path__ = [src_dir]
    sys.modules.setdefault("src", _src_pkg)

    _gfn_pkg = types.ModuleType("src.gflownet")
    _gfn_pkg.__path__ = [src_dir]
    sys.modules.setdefault("src.gflownet", _gfn_pkg)

    try:
        from environment import FaujasiteEnvironment  # noqa: F401 (imported below)
        from model import GFlowNet                    # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Cannot import GFlowNet source: {e}. "
            "Bundle mat-gen/src/gflownet/ into inference/gflownet/gflownet_src/ "
            "before deploying. See the docstring at the top of this file."
        ) from e

    # Re-import for actual use (avoids F821 "undefined name" in linters)
    from environment import FaujasiteEnvironment
    from model import GFlowNet

    # Instantiate environment — use the bundled FAU.cif so we always get the
    # real Faujasite T-site positions (matches training), never the FCC fallback.
    _cif = os.path.join(os.path.dirname(__file__), "FAU.cif")
    env = FaujasiteEnvironment(template_path=_cif if os.path.exists(_cif) else None)

    hidden_dim = int(os.environ.get("GFN_HIDDEN_DIM", "256"))
    num_layers  = int(os.environ.get("GFN_NUM_LAYERS", "3"))

    gfn = GFlowNet(
        env=env,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=str(device),
    )

    # Load checkpoint — state dicts only (optimizer not needed for inference)
    checkpoint_path = _find_checkpoint(model_dir)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    gfn.policy_net.load_state_dict(checkpoint["policy_net"])
    try:
        gfn.flow_net.load_state_dict(checkpoint["flow_net"])
    except RuntimeError:
        # Legacy checkpoint: key was 'log_Z', now 'log_Z_base'
        old_state = checkpoint["flow_net"]
        if "log_Z" in old_state and "log_Z_base" not in old_state:
            old_state["log_Z_base"] = old_state.pop("log_Z")
        gfn.flow_net.load_state_dict(old_state, strict=False)
        logger.info("Loaded legacy flow_net checkpoint (log_Z → log_Z_base)")

    gfn.policy_net.eval()
    gfn.flow_net.eval()

    # Optional fast proxy for reward estimation
    try:
        from fast_proxy import FastPhysicsProxy, ProxyConfig
        fast_proxy = FastPhysicsProxy(ProxyConfig())
        logger.info("FastPhysicsProxy loaded")
    except Exception as e:
        logger.warning(f"FastPhysicsProxy not available: {e}")
        fast_proxy = None

    logger.info(
        f"GFlowNet ready — {env.num_t_sites} T-sites, "
        f"hidden_dim={hidden_dim}, num_layers={num_layers}"
    )


def run(raw_data: str) -> str:
    """Run inference. Called once per request."""
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    mode = data.get("mode", "generate")

    if mode == "generate":
        return _handle_generate(data)
    elif mode == "score":
        return _handle_score(data)
    else:
        return json.dumps({"error": f"Unknown mode '{mode}'. Use 'generate' or 'score'."})


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def _handle_generate(data: dict) -> str:
    """Sample N zeolite configurations from the GFlowNet policy."""
    n_samples  = int(data.get("n_samples", DEFAULT_N_SAMPLES))
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))

    samples = []
    for _ in range(n_samples):
        try:
            # sample_trajectory returns (states, actions, log_probs, terminal_log_prob)
            states, actions, log_probs, terminal_log_prob = gfn.sample_trajectory(
                temperature=temperature
            )
            final_state = states[-1]

            sample = {
                "config": final_state.config.tolist(),
                "n_substitutions": int(final_state.num_substitutions),
                "si_al_ratio": float(env.get_si_al_ratio(final_state)),
                "trajectory_length": len(states),
            }

            if fast_proxy is not None:
                try:
                    result = fast_proxy.evaluate(final_state.config)
                    sample["reward"]       = float(result.get("reward", 0.0))
                    sample["proxy_energy"] = float(result.get("energy", 0.0))
                except Exception as e:
                    logger.warning(f"Proxy evaluation failed: {e}")

            try:
                from ase.io import write as ase_write
                atoms = env.state_to_atoms(final_state)
                buf = io.BytesIO()
                ase_write(buf, atoms, format="cif")
                sample["cif"] = buf.getvalue().decode("utf-8")
            except Exception as e:
                logger.warning(f"CIF generation failed: {e}")

            samples.append(sample)
        except Exception as e:
            logger.error(f"Sample failed: {e}")
            samples.append({"error": str(e)})

    return json.dumps({"mode": "generate", "samples": samples})


def _handle_score(data: dict) -> str:
    """Score a user-supplied Si/Al configuration."""
    if "config" not in data:
        return json.dumps({"error": "Missing 'config' array (binary Si/Al assignment)"})

    config_list = data["config"]
    config = np.array(config_list, dtype=np.float32)

    if len(config) != env.num_t_sites:
        return json.dumps({
            "error": (
                f"Config length {len(config)} doesn't match "
                f"num_t_sites={env.num_t_sites}"
            )
        })

    from environment import FaujasiteState
    n_al = int(config.sum())
    modified = [i for i, v in enumerate(config) if v == 1]
    state = FaujasiteState(
        config=config,
        num_substitutions=n_al,
        modified_sites=modified,
    )

    result = {
        "mode": "score",
        "config": config_list,
        "n_substitutions": n_al,
        "si_al_ratio": float(env.get_si_al_ratio(state)),
    }

    if fast_proxy is not None:
        try:
            proxy_result = fast_proxy.evaluate(config)
            result["reward"]       = float(proxy_result.get("reward", 0.0))
            result["proxy_energy"] = float(proxy_result.get("energy", 0.0))
        except Exception as e:
            result["proxy_error"] = str(e)

    # Policy entropy — how likely GFlowNet is to generate this config
    try:
        state_tensor = torch.tensor(
            env.state_to_tensor(state), dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Build valid-actions mask (required by PolicyNetwork.forward)
        valid_actions = env.get_valid_actions(state)
        num_actions = env.num_t_sites + 1
        mask = torch.zeros(1, num_actions, device=device)
        for a in valid_actions:
            mask[0, a] = 1.0

        with torch.no_grad():
            logits = gfn.policy_net(state_tensor, mask)
        result["policy_entropy"] = float(
            torch.distributions.Categorical(logits=logits).entropy().item()
        )
    except Exception as e:
        logger.warning(f"Policy entropy computation failed: {e}")

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint(model_dir: str) -> str:
    import glob
    import re

    candidates = glob.glob(os.path.join(model_dir, "**/*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under {model_dir}")

    # Prefer highest-iteration checkpoint (gfn_checkpoint_300.pt > gfn_checkpoint_100.pt)
    def _iter(p: str) -> int:
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return max(candidates, key=_iter)
