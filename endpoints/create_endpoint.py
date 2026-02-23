#!/usr/bin/env python3
"""
Create or update an Azure ML Managed Online Endpoint.

One endpoint can serve multiple models via named deployments.
The recommended pattern:
  - One endpoint per model family (e.g. cgcnn-inference, gflownet-inference)
  - Multiple deployments per endpoint (e.g. v1=experimental, v2=production)
  - Traffic split between deployments for A/B testing

Usage:
    # Create endpoint for CGCNN models
    python endpoints/create_endpoint.py --name cgcnn-inference

    # Create endpoint with description and tags
    python endpoints/create_endpoint.py \
        --name gflownet-inference \
        --description "GFlowNet zeolite generator" \
        --tags project=chemia model_type=gflownet

    # Update an existing endpoint (same command, idempotent)
    python endpoints/create_endpoint.py --name cgcnn-inference

    # Delete an endpoint (removes all deployments too)
    python endpoints/create_endpoint.py --name cgcnn-inference --delete

    # List all endpoints
    python endpoints/create_endpoint.py --list
"""

import argparse
import sys


def build_client(args):
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    subscription = args.subscription
    if not subscription:
        import subprocess
        try:
            r = subprocess.run(
                ["az", "account", "show", "--query", "id", "-o", "tsv"],
                capture_output=True, text=True, check=True
            )
            subscription = r.stdout.strip()
        except Exception:
            print("ERROR: Could not get Azure subscription. Run: az login")
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
        print("ERROR: Pass --workspace and --resource-group, or run from a compute instance.")
        sys.exit(1)


def cmd_list(ml_client):
    print("\nManaged Online Endpoints:")
    print(f"{'Name':<40} {'Auth':<10} {'Provisioning':<20} {'Traffic'}")
    print("-" * 90)
    for ep in ml_client.online_endpoints.list():
        traffic = ", ".join(f"{k}={v}%" for k, v in (ep.traffic or {}).items())
        print(
            f"{ep.name:<40} {ep.auth_mode:<10} "
            f"{ep.provisioning_state:<20} {traffic or '(no deployments)'}"
        )


def cmd_create(ml_client, args):
    from azure.ai.ml.entities import ManagedOnlineEndpoint

    tags = {}
    if args.tags:
        for kv in args.tags:
            k, _, v = kv.partition("=")
            tags[k] = v

    endpoint = ManagedOnlineEndpoint(
        name=args.name,
        description=args.description or f"Inference endpoint for {args.name}",
        auth_mode="key",
        tags=tags,
    )

    print(f"Creating/updating endpoint: {args.name} ...")
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    result = poller.result()
    print(f"Done. Provisioning state: {result.provisioning_state}")

    # Print the scoring URI and key retrieval hint
    ep = ml_client.online_endpoints.get(args.name)
    print(f"\nScoring URI: {ep.scoring_uri}")
    print(f"\nGet API key:")
    print(f"  az ml online-endpoint get-credentials \\")
    print(f"    --name {args.name} \\")
    print(f"    --workspace-name {ml_client.workspace_name} \\")
    print(f"    --resource-group {ml_client.resource_group_name}")


def cmd_delete(ml_client, args):
    print(f"Deleting endpoint: {args.name} (and all its deployments) ...")
    confirm = input("Type the endpoint name to confirm: ").strip()
    if confirm != args.name:
        print("Cancelled.")
        return
    ml_client.online_endpoints.begin_delete(name=args.name).result()
    print("Deleted.")


def main():
    parser = argparse.ArgumentParser(
        description="Create or manage Azure ML Managed Online Endpoints"
    )
    parser.add_argument("--name", "-n", help="Endpoint name (e.g. cgcnn-inference)")
    parser.add_argument("--description", help="Endpoint description")
    parser.add_argument(
        "--tags", nargs="+", metavar="KEY=VALUE",
        help="Tags (e.g. project=chemia model_type=cgcnn)"
    )
    parser.add_argument("--workspace", "-w", default=None)
    parser.add_argument("--resource-group", "-g", default=None)
    parser.add_argument("--subscription", "-s", default=None)
    parser.add_argument("--list", action="store_true", help="List all endpoints")
    parser.add_argument("--delete", action="store_true", help="Delete the named endpoint")

    args = parser.parse_args()

    try:
        from azure.ai.ml import MLClient  # noqa: F401
    except ImportError:
        print("ERROR: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    ml_client = build_client(args)

    if args.list:
        cmd_list(ml_client)
    elif args.delete:
        if not args.name:
            print("ERROR: --name required for --delete")
            sys.exit(1)
        cmd_delete(ml_client, args)
    elif args.name:
        cmd_create(ml_client, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
