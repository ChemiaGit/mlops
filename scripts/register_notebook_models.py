#!/usr/bin/env python3
"""
Register CGCNN models from the Notebooks filesystem into Azure ML Model Registry.

Models live under Users/vladimir.kazarin/models_storage/ on the workspace
file share. This script:
  1. Reads local_model_registry.json (if present) for metrics & metadata
  2. Scans models_storage/ folders for .pt checkpoints
  3. Registers each model (latest or all checkpoints) to the Model Registry

Run this from the Azure ML Notebooks terminal or a compute instance
where the workspace file share is mounted.

Usage:
    # From the vladimir.kazarin/ directory in Azure ML Notebooks terminal:

    # Register a specific model folder (name = folder minus timestamp)
    python register_notebook_models.py --folder CGCNN_specific_heat_1757119144
    # → registers as "CGCNN_specific_heat"

    python register_notebook_models.py --folder CGCNN_simple_electronic_zt_n_1757293111
    # → registers as "CGCNN_simple_electronic_zt_n"

    # Dry run
    python register_notebook_models.py --folder CGCNN_specific_heat_1757119144 --dry-run

    # Register all checkpoints (not just latest)
    python register_notebook_models.py --folder CGCNN_specific_heat_1757119144 --all-checkpoints

    # Mark as production-ready
    python register_notebook_models.py --folder CGCNN_specific_heat_1757119144 --stage production

    # Register ALL model folders at once
    python register_notebook_models.py --all

    # Custom models_storage path
    python register_notebook_models.py --folder CGCNN_specific_heat_1757119144 \
        --models-dir ./models_storage
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Filesystem scanning
# ---------------------------------------------------------------------------

def discover_model_folders(models_dir: Path) -> List[Path]:
    """Find all CGCNN_* folders in models_storage/."""
    if not models_dir.exists():
        print(f"ERROR: models_storage dir not found: {models_dir}")
        sys.exit(1)

    folders = sorted([
        d for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith("CGCNN_")
    ])
    return folders


def find_checkpoints(folder: Path) -> List[Path]:
    """
    Find .pt checkpoint files in a model folder.
    Handles both naming conventions:
      - checkpoint1.pt, checkpoint2.pt, ...
      - checkpoint_100.pt, checkpoint_200.pt, ...
      - gfn_checkpoint_100.pt, ...
    Returns sorted by number (ascending).
    """
    pts = sorted(folder.glob("*.pt"))
    if not pts:
        # Check outputs/ subfolder
        pts = sorted((folder / "outputs").glob("*.pt")) if (folder / "outputs").exists() else []

    # Sort numerically
    def _extract_num(p: Path) -> int:
        match = re.search(r'(\d+)', p.stem)
        return int(match.group(1)) if match else 0

    return sorted(pts, key=_extract_num)


def find_training_results(folder: Path) -> Optional[Dict]:
    """Load training_results.json if present."""
    for candidate in [
        folder / "training_results.json",
        folder / "outputs" / "training_results.json",
    ]:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Bad JSON in {candidate}: {e}")
    return None


def find_training_plots(folder: Path) -> Optional[Path]:
    """Find training_plots.png if present."""
    for candidate in [
        folder / "training_plots.png",
        folder / "outputs" / "training_plots.png",
    ]:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def strip_timestamp(folder_name: str) -> str:
    """
    Strip trailing timestamp from folder name to get the model name.
      CGCNN_specific_heat_1757119144       → CGCNN_specific_heat
      CGCNN_simple_electronic_zt_n_1757293111 → CGCNN_simple_electronic_zt_n
      CGCNN_deeper_1725822243              → CGCNN_deeper
    """
    return re.sub(r'_\d{10,}$', '', folder_name)


def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """
    Extract metadata from folder name for tags.
    """
    info = {}
    info["folder_name"] = folder_name
    info["model_name"] = strip_timestamp(folder_name)

    # Extract timestamp
    ts_match = re.search(r'(\d{10,})$', folder_name)
    if ts_match:
        info["job_timestamp"] = ts_match.group(1)

    return info


def build_tags_from_registry_entry(
    entry: Dict, folder_info: Dict, stage: str
) -> Dict[str, str]:
    """Build model tags from a local_model_registry.json entry + folder metadata."""
    tags = {}

    # Add folder-derived metadata
    tags.update({k: str(v) for k, v in folder_info.items()})
    tags["stage"] = stage
    tags["source"] = "notebooks_filesystem"

    # Add all metrics from registry entry (flatten)
    for key, value in entry.items():
        if key == "path":
            tags["source_path"] = str(value)
        elif isinstance(value, (int, float)):
            tags[key] = f"{value:.6f}" if isinstance(value, float) else str(value)
        elif isinstance(value, str):
            tags[key] = value
        elif isinstance(value, dict):
            # Flatten nested dicts: {"metrics": {"mae": 0.1}} -> "metrics_mae": "0.1"
            for sub_key, sub_val in value.items():
                flat_key = f"{key}_{sub_key}"
                if isinstance(sub_val, float):
                    tags[flat_key] = f"{sub_val:.6f}"
                else:
                    tags[flat_key] = str(sub_val)

    return tags


def build_tags_from_results_json(
    results: Dict, folder_info: Dict, stage: str
) -> Dict[str, str]:
    """Build model tags from training_results.json + folder metadata."""
    tags = {}
    tags.update({k: str(v) for k, v in folder_info.items()})
    tags["stage"] = stage
    tags["source"] = "notebooks_filesystem"

    # Flatten training results into tags
    for key, value in results.items():
        if isinstance(value, float):
            tags[key] = f"{value:.6f}"
        elif isinstance(value, (int, str, bool)):
            tags[key] = str(value)
        elif isinstance(value, list):
            # Store list length and last value for things like loss curves
            tags[f"{key}_length"] = str(len(value))
            if value and isinstance(value[-1], (int, float)):
                tags[f"{key}_final"] = f"{value[-1]:.6f}" if isinstance(value[-1], float) else str(value[-1])
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat_key = f"{key}_{sub_key}"
                if isinstance(sub_val, float):
                    tags[flat_key] = f"{sub_val:.6f}"
                else:
                    tags[flat_key] = str(sub_val)

    return tags


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_model(
    ml_client,
    checkpoint_path: Path,
    model_name: str,
    tags: Dict[str, str],
    description: str,
    dry_run: bool = False,
) -> bool:
    """Register a single checkpoint to the Azure ML Model Registry."""
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes

    if dry_run:
        print(f"  [DRY RUN] Would register: {checkpoint_path.name}")
        return True

    model = Model(
        path=str(checkpoint_path),
        name=model_name,
        description=description,
        type=AssetTypes.CUSTOM_MODEL,
        tags=tags,
    )

    try:
        registered = ml_client.models.create_or_update(model)
        print(f"  ✓ Registered: {registered.name} v{registered.version}")
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def process_model_folder(
    ml_client,
    folder: Path,
    registry_entries: Dict,
    stage: str,
    all_checkpoints: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    """Process a single model folder: discover checkpoints, build tags, register."""

    folder_name = folder.name
    folder_info = parse_folder_name(folder_name)
    model_name = folder_info["model_name"]

    print(f"\n  Model name: {model_name}")

    # Find checkpoints
    checkpoints = find_checkpoints(folder)
    if not checkpoints:
        print(f"  No .pt files found, skipping.")
        return 0, 0

    print(f"  Found {len(checkpoints)} checkpoint(s): {[c.name for c in checkpoints]}")

    # Build tags — prefer local_model_registry.json, fall back to training_results.json
    # Try to match this folder to a registry entry
    tags = {}
    registry_key = _match_registry_key(folder_name, registry_entries)
    if registry_key:
        print(f"  Matched registry entry: '{registry_key}'")
        tags = build_tags_from_registry_entry(
            registry_entries[registry_key], folder_info, stage
        )
    else:
        # Fall back to training_results.json
        results = find_training_results(folder)
        if results:
            print(f"  Using training_results.json for metrics")
            tags = build_tags_from_results_json(results, folder_info, stage)
        else:
            print(f"  No metrics found (no registry entry or training_results.json)")
            tags = {k: str(v) for k, v in folder_info.items()}
            tags["stage"] = stage
            tags["source"] = "notebooks_filesystem"

    # Register checkpoints
    registered = 0
    errors = 0

    if all_checkpoints:
        for ckpt in checkpoints:
            ckpt_tags = dict(tags)
            ckpt_num = re.search(r'(\d+)', ckpt.stem)
            ckpt_tags["checkpoint_number"] = ckpt_num.group(1) if ckpt_num else "unknown"

            desc = f"{model_name} checkpoint {ckpt.name}"

            if register_model(ml_client, ckpt, model_name, ckpt_tags, desc, dry_run):
                registered += 1
            else:
                errors += 1
    else:
        # Register only the latest (highest numbered) checkpoint
        latest = checkpoints[-1]
        ckpt_tags = dict(tags)
        ckpt_num = re.search(r'(\d+)', latest.stem)
        ckpt_tags["checkpoint_number"] = ckpt_num.group(1) if ckpt_num else "unknown"
        ckpt_tags["is_latest"] = "true"

        desc = f"{model_name} (latest: {latest.name})"

        if register_model(ml_client, latest, model_name, ckpt_tags, desc, dry_run):
            registered += 1
        else:
            errors += 1

    return registered, errors


def _match_registry_key(folder_name: str, registry: Dict) -> Optional[str]:
    """
    Try to match a folder name to a key in local_model_registry.json.
    Handles partial matches (e.g. 'specific_heat' matches folder
    'CGCNN_specific_heat_1757119144').
    """
    if not registry:
        return None

    folder_lower = folder_name.lower()
    for key in registry:
        # Exact path match
        entry = registry[key]
        if isinstance(entry, dict) and "path" in entry:
            if folder_name in str(entry["path"]):
                return key

        # Key name is substring of folder name
        if key.lower().replace(" ", "_") in folder_lower:
            return key

        # Folder name contains the key
        if isinstance(entry, dict) and "name" in entry:
            if entry["name"].lower().replace(" ", "_") in folder_lower:
                return key

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Register CGCNN models from Notebooks filesystem to Azure ML Model Registry"
    )

    # What to register
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--folder", type=str,
        help="Folder name to register (e.g. CGCNN_specific_heat_1757119144)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Register all CGCNN_* folders in models_storage/"
    )

    # Paths
    parser.add_argument(
        "--models-dir", type=Path, default=Path("./models_storage"),
        help="Path to models_storage/ folder (default: ./models_storage)"
    )
    parser.add_argument(
        "--registry-json", type=Path, default=Path("./local_model_registry.json"),
        help="Path to local_model_registry.json (default: ./local_model_registry.json)"
    )

    # Azure ML connection (optional — auto-detected on compute instances)
    parser.add_argument("--workspace", "-w", default=None)
    parser.add_argument("--resource-group", "-g", default=None)
    parser.add_argument("--subscription", "-s", default=None)

    # Registration options
    parser.add_argument(
        "--all-checkpoints", action="store_true",
        help="Register all checkpoints, not just the latest"
    )
    parser.add_argument(
        "--stage", default="experimental",
        choices=["experimental", "staging", "production", "archived"],
        help="Model stage tag (default: experimental)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be registered without doing it")

    args = parser.parse_args()

    # Load local registry if available
    registry_entries = {}
    if args.registry_json.exists():
        print(f"Loading registry: {args.registry_json}")
        with open(args.registry_json) as f:
            registry_entries = json.load(f)
        print(f"  Found {len(registry_entries)} entries")
    else:
        print(f"No local_model_registry.json found at {args.registry_json}")
        print(f"  Will use training_results.json from each folder instead")

    # Resolve folders
    if args.folder:
        folder_path = args.models_dir / args.folder
        if not folder_path.exists():
            print(f"ERROR: Folder not found: {folder_path}")
            sys.exit(1)
        folders = [folder_path]
    else:
        folders = discover_model_folders(args.models_dir)

    print(f"\nProcessing {len(folders)} folder(s)")

    if not folders:
        print("No model folders to process.")
        return

    # Connect to Azure ML
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("ERROR: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    if args.workspace and args.resource_group:
        subscription = args.subscription
        if not subscription:
            import subprocess
            try:
                result = subprocess.run(
                    ["az", "account", "show", "--query", "id", "-o", "tsv"],
                    capture_output=True, text=True, check=True
                )
                subscription = result.stdout.strip()
            except Exception:
                print("ERROR: Pass --subscription or run: az login")
                sys.exit(1)

        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace,
        )
    else:
        # Auto-detect from compute instance
        try:
            ml_client = MLClient.from_config(
                credential=DefaultAzureCredential()
            )
            print(f"Auto-detected workspace: {ml_client.workspace_name}")
        except Exception:
            print("ERROR: Could not auto-detect workspace.")
            print("  Either run on a compute instance, or pass --workspace and --resource-group")
            sys.exit(1)

    # Process each folder
    total_registered = 0
    total_skipped = 0
    total_errors = 0

    for folder in folders:
        print(f"\n{'='*60}")
        print(f"📁 {folder.name}")
        print(f"{'='*60}")

        registered, errors = process_model_folder(
            ml_client=ml_client,
            folder=folder,
            registry_entries=registry_entries,
            stage=args.stage,
            all_checkpoints=args.all_checkpoints,
            dry_run=args.dry_run,
        )

        total_registered += registered
        total_errors += errors
        if registered == 0 and errors == 0:
            total_skipped += 1

    # Summary
    action = "Would register" if args.dry_run else "Registered"
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Folders scanned: {len(folders)}")
    print(f"  {action}: {total_registered} checkpoints")
    print(f"  Skipped: {total_skipped} folders (no checkpoints)")
    print(f"  Errors: {total_errors}")

    if args.dry_run and total_registered > 0:
        print(f"\nRe-run without --dry-run to register these models.")

    # Print registry overview
    if not args.dry_run and total_registered > 0:
        print(f"\n📋 Registered models (query with):")
        print(f"  az ml model list --workspace-name <ws> --resource-group <rg> "
              f"--query \"[?tags.model_family=='cgcnn']\"")


if __name__ == "__main__":
    main()