#!/usr/bin/env python3
"""
Retroactively register .pt checkpoints from completed Azure ML jobs.

Scans past jobs in an experiment, finds .pt outputs, extracts training
parameters from the job command, and registers them to the Model Registry.

Usage:
    # Dry run (list what would be registered)
    python scripts/register_existing_models.py \
        --workspace <workspace> \
        --resource-group <rg> \
        --dry-run

    # Register final checkpoints from all jobs (auto-discovers)
    python scripts/register_existing_models.py \
        --workspace <workspace> \
        --resource-group <rg>

    # Register from specific jobs with known checkpoint iterations
    python scripts/register_existing_models.py \
        --workspace <workspace> \
        --resource-group <rg> \
        --job-ids quiet_sock_qqcd596hy9 \
        --checkpoint-iterations 300

    # Register all available checkpoints (not just final)
    python scripts/register_existing_models.py \
        --workspace <workspace> \
        --resource-group <rg> \
        --all-checkpoints
"""

import argparse
import re
import sys
from typing import Dict, List, Optional


def parse_training_params_from_command(command: str) -> Dict[str, str]:
    """
    Extract training hyperparameters from the job's command string.

    Parses flags like --num_iterations 500 --batch_size 16 into a dict.
    """
    params = {}

    # Match --flag value patterns (value is non-flag token)
    flag_pattern = re.compile(r'--([\w_-]+)\s+([^\s-][^\s]*)')
    for match in flag_pattern.finditer(command):
        key = match.group(1).replace("-", "_")
        value = match.group(2)
        params[key] = value

    # Match boolean flags (--mock_oracle, --spot)
    bool_pattern = re.compile(r'--([\w_-]+)(?=\s+--|$|\s*&&)')
    for match in bool_pattern.finditer(command):
        key = match.group(1).replace("-", "_")
        if key not in params:
            params[key] = "true"

    return params


def _get_save_interval_from_command(command: str) -> int:
    """Extract save_interval from command, default 100."""
    match = re.search(r'--save_interval\s+(\d+)', command)
    return int(match.group(1)) if match else 100


def _get_num_iterations_from_command(command: str) -> Optional[int]:
    """Extract num_iterations from command."""
    match = re.search(r'--num_iterations\s+(\d+)', command)
    return int(match.group(1)) if match else None


def _build_tags(job, stage: str) -> Dict[str, str]:
    """Build tags dict from a job's command and metadata."""
    command = getattr(job, 'command', '') or ''
    params = parse_training_params_from_command(command)

    tags = {k: str(v) for k, v in params.items()}
    tags["source_job_id"] = job.name
    tags["source_job_display_name"] = getattr(job, 'display_name', '') or ''
    tags["source_job_status"] = getattr(job, 'status', '') or ''
    tags["source_experiment"] = getattr(job, 'experiment_name', '') or ''
    tags["registered_retroactively"] = "true"
    tags["stage"] = stage

    if hasattr(job, 'creation_context') and job.creation_context:
        created = getattr(job.creation_context, 'created_at', None)
        if created:
            tags["job_created_at"] = str(created)

    return tags


def try_register_checkpoint(
    ml_client,
    job,
    checkpoint_path: str,
    model_name: str,
    tags: Dict[str, str],
) -> bool:
    """
    Try to register a single checkpoint. Returns True if successful,
    False if the artifact doesn't exist.
    Raises on unexpected errors.
    """
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes

    artifact_uri = f"azureml://jobs/{job.name}/outputs/artifacts/paths/{checkpoint_path}"

    iter_match = re.search(r'checkpoint_(\d+)\.pt', checkpoint_path)
    iteration = iter_match.group(1) if iter_match else "unknown"

    checkpoint_tags = dict(tags)
    checkpoint_tags["checkpoint_iteration"] = iteration

    model = Model(
        path=artifact_uri,
        name=model_name,
        description=f"GFlowNet checkpoint (iter {iteration}) from job {job.name}",
        type=AssetTypes.CUSTOM_MODEL,
        tags=checkpoint_tags,
    )

    try:
        registered = ml_client.models.create_or_update(model)
        print(f"  Registered: {registered.name} v{registered.version} (iter {iteration})")
        return True
    except Exception as e:
        error_str = str(e)
        if "NoMatchingArtifactsFoundFromJob" in error_str:
            return False
        raise


def register_job(
    ml_client,
    job,
    model_name: str,
    stage: str = "experimental",
    all_checkpoints: bool = False,
    checkpoint_iterations: Optional[List[int]] = None,
    dry_run: bool = False,
) -> tuple:
    """
    Register checkpoints from a single job.

    Discovery strategy:
    1. If --checkpoint-iterations given, use those exactly.
    2. Otherwise, try registering candidates (every save_interval up to
       num_iterations). Failures due to missing artifacts are silently
       skipped - only real artifacts get registered.

    Returns (registered_count, error_count).
    """
    command = getattr(job, 'command', '') or ''
    tags = _build_tags(job, stage)

    # Determine which iterations to attempt
    if checkpoint_iterations:
        candidates = checkpoint_iterations
    else:
        num_iterations = _get_num_iterations_from_command(command)
        save_interval = _get_save_interval_from_command(command)

        if not num_iterations:
            print(f"  Could not determine num_iterations from command.")
            print(f"  Use --checkpoint-iterations to specify manually.")
            return 0, 0

        # Candidates: every save_interval + the final iteration
        candidates = list(range(save_interval, num_iterations + 1, save_interval))
        if num_iterations not in candidates:
            candidates.append(num_iterations)

    if dry_run:
        print(f"  [DRY RUN] Would attempt {len(candidates)} checkpoint(s):")
        for it in candidates:
            print(f"    outputs/gfn_checkpoint_{it}.pt")
        print(f"  Tags ({len(tags)}):")
        for k, v in sorted(tags.items()):
            print(f"    {k}: {v}")
        return len(candidates), 0

    # Try registering each candidate; skip missing ones
    registered = 0
    errors = 0

    if all_checkpoints:
        # Try all candidates, skip missing
        print(f"  Trying {len(candidates)} checkpoint candidates...")
        for iteration in candidates:
            ckpt_path = f"outputs/gfn_checkpoint_{iteration}.pt"
            try:
                if try_register_checkpoint(ml_client, job, ckpt_path, model_name, tags):
                    registered += 1
            except Exception as e:
                print(f"  ERROR on iter {iteration}: {e}")
                errors += 1
    else:
        # Find the highest existing checkpoint (try from top down)
        print(f"  Finding latest checkpoint (trying {len(candidates)} candidates from highest)...")
        for iteration in reversed(candidates):
            ckpt_path = f"outputs/gfn_checkpoint_{iteration}.pt"
            try:
                if try_register_checkpoint(ml_client, job, ckpt_path, model_name, tags):
                    registered += 1
                    break  # Only register the latest
            except Exception as e:
                print(f"  ERROR on iter {iteration}: {e}")
                errors += 1
                break

        if registered == 0 and errors == 0:
            print(f"  No checkpoints found in job outputs.")

    return registered, errors


def main():
    parser = argparse.ArgumentParser(
        description="Register .pt checkpoints from past Azure ML jobs"
    )
    parser.add_argument("--workspace", "-w", required=True)
    parser.add_argument("--resource-group", "-g", required=True)
    parser.add_argument("--subscription", "-s", default=None)
    parser.add_argument("--experiment-name", default="gflownet-bfgs-parallel",
                        help="Experiment to scan (default: gflownet-bfgs-parallel)")
    parser.add_argument("--model-name", default="gfn-zeolite-model",
                        help="Model name in registry (default: gfn-zeolite-model)")
    parser.add_argument("--job-ids", nargs="+", default=None,
                        help="Specific job IDs to register (default: all completed jobs)")
    parser.add_argument("--checkpoint-iterations", type=int, nargs="+", default=None,
                        help="Specific checkpoint iterations to register "
                             "(e.g. --checkpoint-iterations 100 200 300)")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Register all available checkpoints, not just the final one")
    parser.add_argument("--stage", default="experimental",
                        help="Model stage tag (default: experimental)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be registered without doing it")
    parser.add_argument("--status-filter", default="All",
                        choices=["Completed", "Canceled", "Failed", "All"],
                        help="Filter jobs by status (default: All)")

    args = parser.parse_args()

    # Import Azure ML SDK
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("ERROR: Install azure-ai-ml: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    # Get subscription
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
            print("ERROR: Could not get subscription. Pass --subscription or run: az login")
            sys.exit(1)

    # Connect
    print(f"Connecting to workspace: {args.workspace}")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    # List jobs
    if args.job_ids:
        print(f"\nFetching {len(args.job_ids)} specified jobs...")
        jobs = []
        for job_id in args.job_ids:
            try:
                job = ml_client.jobs.get(job_id)
                jobs.append(job)
            except Exception as e:
                print(f"  WARNING: Could not fetch job {job_id}: {e}")
    else:
        print(f"\nListing jobs from experiment: {args.experiment_name}")
        try:
            all_jobs = list(ml_client.jobs.list(experiment_name=args.experiment_name))
        except Exception:
            print("  Experiment filter failed, listing all jobs...")
            all_jobs = list(ml_client.jobs.list())

        if args.status_filter != "All":
            jobs = [j for j in all_jobs if getattr(j, 'status', '') == args.status_filter]
        else:
            jobs = all_jobs

        print(f"  Found {len(jobs)} jobs (status={args.status_filter})")

    if not jobs:
        print("No jobs found. Nothing to register.")
        return

    # Process each job
    total_registered = 0
    total_skipped = 0
    total_errors = 0

    for job in jobs:
        job_name = job.name
        status = getattr(job, 'status', 'unknown')
        display = getattr(job, 'display_name', '') or job_name
        print(f"\n{'='*60}")
        print(f"Job: {display} ({job_name}) [{status}]")
        print(f"{'='*60}")

        registered, errors = register_job(
            ml_client=ml_client,
            job=job,
            model_name=args.model_name,
            stage=args.stage,
            all_checkpoints=args.all_checkpoints,
            checkpoint_iterations=args.checkpoint_iterations,
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
    print(f"  {action}: {total_registered} checkpoints")
    print(f"  Skipped: {total_skipped} jobs (no checkpoints found)")
    print(f"  Errors: {total_errors}")

    if args.dry_run and total_registered > 0:
        print(f"\nRe-run without --dry-run to register these models.")


if __name__ == "__main__":
    main()
