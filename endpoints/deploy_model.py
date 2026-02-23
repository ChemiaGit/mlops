#!/usr/bin/env python3
"""
Deploy a registered Azure ML model to a Managed Online Endpoint.

Workflow:
  1. A model is registered (via register_notebook_models.py or
     register_existing_models.py) with a name like CGCNN_specific_heat v2.
  2. This script creates a deployment that binds that model version
     to a running endpoint, pointing at the scoring script in inference/.
  3. Traffic is shifted to the new deployment when ready.

Usage:
    # Deploy CGCNN specific heat model v2 to cgcnn-inference endpoint
    python endpoints/deploy_model.py \
        --endpoint cgcnn-inference \
        --model "CGCNN_specific_heat:2" \
        --deployment-name v2 \
        --scoring-dir ./inference/cgcnn \
        --instance-type Standard_NC4as_T4_v3

    # Deploy GFlowNet model
    python endpoints/deploy_model.py \
        --endpoint gflownet-inference \
        --model "gfn-zeolite-model:5" \
        --deployment-name v1 \
        --scoring-dir ./inference/gflownet \
        --instance-type Standard_NC4as_T4_v3

    # Deploy on CPU (cheaper, for CGCNN with small structures)
    python endpoints/deploy_model.py \
        --endpoint cgcnn-inference \
        --model "CGCNN_specific_heat:2" \
        --deployment-name v2-cpu \
        --scoring-dir ./inference/cgcnn \
        --instance-type Standard_DS3_v2

    # Shift 100% traffic to a deployment
    python endpoints/deploy_model.py \
        --endpoint cgcnn-inference \
        --set-traffic v2=100

    # Split traffic 80/20 between deployments (A/B test)
    python endpoints/deploy_model.py \
        --endpoint cgcnn-inference \
        --set-traffic v2=80 v1=20

    # List deployments on an endpoint
    python endpoints/deploy_model.py --endpoint cgcnn-inference --list

    # Delete a deployment
    python endpoints/deploy_model.py \
        --endpoint cgcnn-inference \
        --deployment-name v1 \
        --delete

Instance type reference:
    Standard_NC4as_T4_v3    4 vCPU,  28 GB,  T4 GPU  ~$0.53/hr
    Standard_NC8as_T4_v3    8 vCPU,  56 GB,  T4 GPU  ~$0.75/hr
    Standard_DS3_v2         4 vCPU,  14 GB,  no GPU  ~$0.20/hr  (CPU only)
    Standard_DS4_v2         8 vCPU,  28 GB,  no GPU  ~$0.40/hr  (CPU only)
"""

import argparse
import os
import sys


# Azure ML curated GPU environment (pytorch + cuda)
DEFAULT_ENVIRONMENT = "AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10"
DEFAULT_INSTANCE_TYPE = "Standard_NC4as_T4_v3"
DEFAULT_INSTANCE_COUNT = 1


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


def cmd_list(ml_client, endpoint_name: str):
    print(f"\nDeployments on '{endpoint_name}':")
    print(f"{'Name':<30} {'Model':<40} {'Instance':<28} {'State'}")
    print("-" * 110)
    for d in ml_client.online_deployments.list(endpoint_name=endpoint_name):
        model_str = str(d.model) if d.model else "—"
        instance = getattr(d, "instance_type", "—")
        print(
            f"{d.name:<30} {model_str:<40} {instance:<28} "
            f"{getattr(d, 'provisioning_state', '—')}"
        )
    ep = ml_client.online_endpoints.get(endpoint_name)
    traffic = ep.traffic or {}
    print(f"\nTraffic: {traffic or '(none)'}")


def cmd_deploy(ml_client, args):
    from azure.ai.ml.entities import (
        CodeConfiguration,
        ManagedOnlineDeployment,
    )

    scoring_dir = os.path.abspath(args.scoring_dir)
    if not os.path.isdir(scoring_dir):
        print(f"ERROR: scoring-dir not found: {scoring_dir}")
        sys.exit(1)
    score_py = os.path.join(scoring_dir, "score.py")
    if not os.path.isfile(score_py):
        print(f"ERROR: score.py not found in {scoring_dir}")
        sys.exit(1)

    model_ref = f"azureml:{args.model}"
    environment = args.environment or DEFAULT_ENVIRONMENT

    env_vars = {}
    if args.env_vars:
        for kv in args.env_vars:
            k, _, v = kv.partition("=")
            env_vars[k] = v

    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint,
        model=model_ref,
        code_configuration=CodeConfiguration(
            code=scoring_dir,
            scoring_script="score.py",
        ),
        environment=environment,
        instance_type=args.instance_type or DEFAULT_INSTANCE_TYPE,
        instance_count=args.instance_count or DEFAULT_INSTANCE_COUNT,
        environment_variables=env_vars or None,
    )

    print(f"Deploying model '{args.model}' to endpoint '{args.endpoint}' "
          f"as deployment '{args.deployment_name}' ...")
    print(f"  Environment:   {environment}")
    print(f"  Instance type: {deployment.instance_type}")
    print(f"  Instance count: {deployment.instance_count}")
    print(f"  Scoring dir:   {scoring_dir}")
    if env_vars:
        print(f"  Env vars:      {env_vars}")

    poller = ml_client.online_deployments.begin_create_or_update(deployment)
    result = poller.result()
    print(f"Done. Provisioning state: {result.provisioning_state}")

    # If this is the first deployment, auto-route 100% traffic to it
    ep = ml_client.online_endpoints.get(args.endpoint)
    if not ep.traffic:
        print(f"\nNo traffic yet — routing 100% to '{args.deployment_name}'")
        ep.traffic = {args.deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(ep).result()
        print("Traffic updated.")
    else:
        print(f"\nCurrent traffic: {ep.traffic}")
        print(
            f"To route traffic to this deployment:\n"
            f"  python endpoints/deploy_model.py "
            f"--endpoint {args.endpoint} "
            f"--set-traffic {args.deployment_name}=100"
        )


def cmd_set_traffic(ml_client, endpoint_name: str, traffic_args: list):
    """Set traffic split between deployments."""
    traffic = {}
    for kv in traffic_args:
        name, _, pct = kv.partition("=")
        traffic[name] = int(pct)

    total = sum(traffic.values())
    if total != 100:
        print(f"ERROR: Traffic percentages must sum to 100, got {total}")
        sys.exit(1)

    ep = ml_client.online_endpoints.get(endpoint_name)
    ep.traffic = traffic
    ml_client.online_endpoints.begin_create_or_update(ep).result()
    print(f"Traffic updated: {traffic}")


def cmd_delete_deployment(ml_client, endpoint_name: str, deployment_name: str):
    print(f"Deleting deployment '{deployment_name}' from '{endpoint_name}' ...")

    # Check that this deployment has 0% traffic before deleting
    ep = ml_client.online_endpoints.get(endpoint_name)
    traffic = ep.traffic or {}
    if traffic.get(deployment_name, 0) > 0:
        print(
            f"WARNING: Deployment '{deployment_name}' has {traffic[deployment_name]}% traffic. "
            "Reassign traffic first with --set-traffic."
        )
        confirm = input("Delete anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    ml_client.online_deployments.begin_delete(
        name=deployment_name,
        endpoint_name=endpoint_name,
    ).result()
    print("Deleted.")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy a registered model to an Azure ML Managed Online Endpoint"
    )

    # Endpoint selection
    parser.add_argument("--endpoint", "-e", required=True, help="Endpoint name")

    # Deployment config
    parser.add_argument("--deployment-name", "-d", default="v1",
                        help="Deployment name (default: v1)")
    parser.add_argument(
        "--model", "-m",
        help="Registered model ref (name:version), e.g. CGCNN_specific_heat:2"
    )
    parser.add_argument(
        "--scoring-dir",
        help="Directory containing score.py (e.g. ./inference/cgcnn)"
    )
    parser.add_argument(
        "--environment",
        help=f"Azure ML environment (default: {DEFAULT_ENVIRONMENT})"
    )
    parser.add_argument(
        "--instance-type", default=DEFAULT_INSTANCE_TYPE,
        help=f"VM SKU (default: {DEFAULT_INSTANCE_TYPE})"
    )
    parser.add_argument(
        "--instance-count", type=int, default=DEFAULT_INSTANCE_COUNT,
        help=f"Number of instances (default: {DEFAULT_INSTANCE_COUNT})"
    )
    parser.add_argument(
        "--env-vars", nargs="+", metavar="KEY=VALUE",
        help="Environment variables passed to scoring script (e.g. MODEL_PROPERTY=specific_heat)"
    )

    # Traffic
    parser.add_argument(
        "--set-traffic", nargs="+", metavar="DEPLOYMENT=PCT",
        help="Set traffic split (e.g. --set-traffic v2=80 v1=20)"
    )

    # Actions
    parser.add_argument("--list", action="store_true", help="List deployments")
    parser.add_argument("--delete", action="store_true", help="Delete the named deployment")

    # Azure connection
    parser.add_argument("--workspace", "-w", default=None)
    parser.add_argument("--resource-group", "-g", default=None)
    parser.add_argument("--subscription", "-s", default=None)

    args = parser.parse_args()

    try:
        from azure.ai.ml import MLClient  # noqa: F401
    except ImportError:
        print("ERROR: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    ml_client = build_client(args)

    if args.list:
        cmd_list(ml_client, args.endpoint)
    elif args.set_traffic:
        cmd_set_traffic(ml_client, args.endpoint, args.set_traffic)
    elif args.delete:
        cmd_delete_deployment(ml_client, args.endpoint, args.deployment_name)
    elif args.model and args.scoring_dir:
        cmd_deploy(ml_client, args)
    else:
        parser.print_help()
        print(
            "\nExamples:\n"
            "  # Deploy CGCNN model\n"
            "  python endpoints/deploy_model.py \\\n"
            "    --endpoint cgcnn-inference \\\n"
            "    --model CGCNN_specific_heat:2 \\\n"
            "    --scoring-dir ./inference/cgcnn\n\n"
            "  # Deploy GFlowNet\n"
            "  python endpoints/deploy_model.py \\\n"
            "    --endpoint gflownet-inference \\\n"
            "    --model gfn-zeolite-model:5 \\\n"
            "    --scoring-dir ./inference/gflownet\n"
        )


if __name__ == "__main__":
    main()
