#!/usr/bin/env python3
"""
Register a custom Azure ML environment for GFlowNet inference.

Extends the curated AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu image
with ase (needed for Faujasite structure handling in the scoring script).

Run once before deploying:
    python endpoints/create_gflownet_environment.py \\
        --workspace gflownet-dev-001 \\
        --resource-group baghsev-rg

Then use --environment gflownet-inference-env:1 when deploying.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ENV_NAME    = "gflownet-inference-env"
ENV_VERSION = "1"


def build_client(args):
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    subscription = args.subscription
    if not subscription:
        try:
            r = subprocess.run(
                ["az", "account", "show", "--query", "id", "-o", "tsv"],
                capture_output=True, text=True, check=True,
            )
            subscription = r.stdout.strip()
        except Exception:
            print("ERROR: Run az login first or pass --subscription")
            sys.exit(1)

    if args.workspace and args.resource_group:
        return MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace,
        )
    try:
        client = MLClient.from_config(credential=DefaultAzureCredential())
        print(f"Auto-detected workspace: {client.workspace_name}")
        return client
    except Exception:
        print("ERROR: Pass --workspace / --resource-group or run from a compute instance.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Register gflownet-inference-env in Azure ML"
    )
    parser.add_argument("--workspace", "-w", default=None)
    parser.add_argument("--resource-group", "-g", default=None)
    parser.add_argument("--subscription", "-s", default=None)
    parser.add_argument(
        "--version", default=ENV_VERSION,
        help=f"Environment version (default: {ENV_VERSION})"
    )
    args = parser.parse_args()

    try:
        from azure.ai.ml import MLClient  # noqa: F401
        from azure.ai.ml.entities import Environment  # noqa: F401
    except ImportError:
        print("ERROR: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    ml_client = build_client(args)

    from azure.ai.ml.entities import BuildContext, Environment

    docker_dir = Path(__file__).parent.parent / "inference" / "gflownet"
    print(f"Building from Dockerfile in: {docker_dir}")

    env = Environment(
        name=ENV_NAME,
        version=args.version,
        description="GFlowNet inference: PyTorch 1.13 (ACPT) + ase",
        build=BuildContext(path=str(docker_dir)),
    )

    print(f"\nRegistering environment '{ENV_NAME}' v{args.version} ...")
    registered = ml_client.environments.create_or_update(env)
    print(f"Done. ID: {registered.id}")
    print(f"\nUse when deploying:")
    print(f"  python endpoints/deploy_model.py \\")
    print(f"    --endpoint gflownet-inference \\")
    print(f"    --model gfn-zeolite-model:6 \\")
    print(f"    --deployment-name v1 \\")
    print(f"    --scoring-dir ./inference/gflownet \\")
    print(f"    --environment {ENV_NAME}:{args.version} \\")
    print(f"    --env-vars GFN_HIDDEN_DIM=256 GFN_NUM_LAYERS=3 \\")
    print(f"    --workspace {ml_client.workspace_name} \\")
    print(f"    --resource-group {ml_client.resource_group_name}")


if __name__ == "__main__":
    main()
