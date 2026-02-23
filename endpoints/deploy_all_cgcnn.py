#!/usr/bin/env python3
"""
Deploy all registered CGCNN models to a single cgcnn-inference endpoint.

Option A layout:
    cgcnn-inference endpoint
      ├── specific-heat   → CGCNN_specific_heat:1
      ├── t-debye         → CGCNN_t_debye:1
      ├── dielectric      → CGCNN_dielectric_constant:1
      ├── seebeck-n       → CGCNN_thermoelectric_seebeck_n:1  (v1 best: loss 11.37)
      ├── seebeck-p       → CGCNN_thermoelectric_seebeck_p:1  (v1 best: loss 9.52)
      ├── zt-n            → CGCNN_thermoelectric_electronic_zt_n:1
      └── zt-p            → CGCNN_thermoelectric_electronic_zt_p:1

Traffic routing:
    - Default (no header) → specific-heat (100%)
    - Named routing via header: azureml-model-deployment: <deployment-name>

Usage:
    # Full deploy (create endpoint + all 7 deployments)
    python endpoints/deploy_all_cgcnn.py

    # Dry run — print what would be deployed without doing it
    python endpoints/deploy_all_cgcnn.py --dry-run

    # Only create the endpoint, skip deployments
    python endpoints/deploy_all_cgcnn.py --endpoint-only

    # Deploy a subset of models
    python endpoints/deploy_all_cgcnn.py --only specific-heat t-debye

    # Redeploy one model (e.g. after promoting a new version)
    python endpoints/deploy_all_cgcnn.py --only seebeck-n

    # Use CPU instances (cheaper for small batch workloads)
    python endpoints/deploy_all_cgcnn.py --instance-type Standard_DS3_v2

    # Custom workspace
    python endpoints/deploy_all_cgcnn.py \\
        --workspace bandstructurelearning \\
        --resource-group baghsev-rg
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Deployment manifest
# Each entry: (deployment_name, registered_model_ref, MODEL_PROPERTY env var)
# Model ref format: "name:version"
# Version choices for seebeck/zt: v1 has the best validation loss
# ---------------------------------------------------------------------------
DEPLOYMENTS = [
    ("specific-heat",  "CGCNN_specific_heat:1",                        "specific_heat"),
    ("t-debye",        "CGCNN_t_debye:1",                              "t_debye"),
    ("dielectric",     "CGCNN_dielectric_constant:1",                  "dielectric_constant"),
    ("seebeck-n",      "CGCNN_thermoelectric_seebeck_n:1",             "seebeck_n"),
    ("seebeck-p",      "CGCNN_thermoelectric_seebeck_p:1",             "seebeck_p"),
    ("zt-n",           "CGCNN_thermoelectric_electronic_zt_n:1",       "electronic_zt_n"),
    ("zt-p",           "CGCNN_thermoelectric_electronic_zt_p:1",       "electronic_zt_p"),
]

ENDPOINT_NAME = "cgcnn-inference"
DEFAULT_INSTANCE_TYPE = "Standard_NC4as_T4_v3"
DEFAULT_ENVIRONMENT = "cgcnn-inference-env:2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_client(args):
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    subscription = getattr(args, "subscription", None)
    if not subscription:
        import subprocess
        try:
            r = subprocess.run(
                ["az", "account", "show", "--query", "id", "-o", "tsv"],
                capture_output=True, text=True, check=True,
            )
            subscription = r.stdout.strip()
        except Exception:
            print("ERROR: Cannot get subscription. Run: az login")
            sys.exit(1)

    workspace = getattr(args, "workspace", None)
    rg = getattr(args, "resource_group", None)

    if workspace and rg:
        return MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription,
            resource_group_name=rg,
            workspace_name=workspace,
        )
    try:
        client = MLClient.from_config(credential=DefaultAzureCredential())
        print(f"Auto-detected workspace: {client.workspace_name}")
        return client
    except Exception:
        print("ERROR: Pass --workspace / --resource-group or run from a compute instance.")
        sys.exit(1)


def ensure_endpoint(ml_client, dry_run: bool) -> str:
    """Create endpoint if it doesn't exist. Returns scoring URI."""
    from azure.ai.ml.entities import ManagedOnlineEndpoint

    try:
        ep = ml_client.online_endpoints.get(ENDPOINT_NAME)
        print(f"Endpoint '{ENDPOINT_NAME}' already exists ({ep.provisioning_state})")
        return ep.scoring_uri
    except Exception:
        pass  # doesn't exist yet

    print(f"\nCreating endpoint '{ENDPOINT_NAME}' ...")
    if dry_run:
        print("  [DRY RUN] skipped")
        return "(dry-run)"

    ep = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="CGCNN property prediction — all models",
        auth_mode="key",
        tags={"project": "chemia", "model_type": "cgcnn"},
    )
    result = ml_client.online_endpoints.begin_create_or_update(ep).result()
    print(f"  Created. Provisioning: {result.provisioning_state}")
    return result.scoring_uri


def deploy_one(ml_client, deployment_name: str, model_ref: str, property_name: str,
               scoring_dir: str, instance_type: str, environment: str,
               dry_run: bool) -> bool:
    """Create or update one deployment. Returns True on success."""
    from azure.ai.ml.entities import CodeConfiguration, ManagedOnlineDeployment

    print(f"\n  [{deployment_name}]  model={model_ref}  property={property_name}")

    if dry_run:
        print(f"    [DRY RUN] Would deploy {model_ref} → {deployment_name}")
        return True

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=ENDPOINT_NAME,
        model=f"azureml:{model_ref}",
        code_configuration=CodeConfiguration(
            code=scoring_dir,
            scoring_script="score.py",
        ),
        environment=environment,
        instance_type=instance_type,
        instance_count=1,
        environment_variables={"MODEL_PROPERTY": property_name},
    )

    try:
        result = ml_client.online_deployments.begin_create_or_update(deployment).result()
        print(f"    Done. State: {result.provisioning_state}")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def set_default_traffic(ml_client, default_deployment: str, dry_run: bool):
    """Route 100% of unspecified-header traffic to the default deployment."""
    print(f"\nSetting default traffic → {default_deployment} (100%)")
    if dry_run:
        print("  [DRY RUN] skipped")
        return
    ep = ml_client.online_endpoints.get(ENDPOINT_NAME)
    ep.traffic = {default_deployment: 100}
    ml_client.online_endpoints.begin_create_or_update(ep).result()
    print("  Traffic updated.")


def print_summary(ml_client, scoring_uri: str, succeeded: list, failed: list):
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Endpoint:  {ENDPOINT_NAME}")
    print(f"  URI:       {scoring_uri}")
    print(f"  Succeeded: {len(succeeded)} — {succeeded}")
    if failed:
        print(f"  Failed:    {len(failed)} — {failed}")

    print("\nCall a specific model:")
    print('  curl -X POST <uri>/score \\')
    print('    -H "Authorization: Bearer <key>" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -H "azureml-model-deployment: seebeck-n" \\')
    print('    -d @sample_inputs/cgcnn_sample.json')

    print("\nPython SDK:")
    print("  ml_client.online_endpoints.invoke(")
    print(f'    endpoint_name="{ENDPOINT_NAME}",')
    print('    deployment_name="seebeck-n",   # or specific-heat, t-debye, ...')
    print('    request_file="sample_inputs/cgcnn_sample.json",')
    print("  )")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deploy all CGCNN models to cgcnn-inference endpoint (Option A)"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="DEPLOYMENT",
        help="Deploy only these named deployments (e.g. --only seebeck-n zt-n)"
    )
    parser.add_argument(
        "--instance-type", default=DEFAULT_INSTANCE_TYPE,
        help=f"VM SKU (default: {DEFAULT_INSTANCE_TYPE}). "
             "Use Standard_DS3_v2 for CPU-only (cheaper)."
    )
    parser.add_argument(
        "--environment", default=DEFAULT_ENVIRONMENT,
        help=f"Azure ML curated environment (default: {DEFAULT_ENVIRONMENT})"
    )
    parser.add_argument("--workspace", "-w", default=None)
    parser.add_argument("--resource-group", "-g", default=None)
    parser.add_argument("--subscription", "-s", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deployed without doing it")
    parser.add_argument("--endpoint-only", action="store_true",
                        help="Only create the endpoint, skip all deployments")

    args = parser.parse_args()

    try:
        from azure.ai.ml import MLClient  # noqa: F401
    except ImportError:
        print("ERROR: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    scoring_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "inference", "cgcnn")
    )
    if not os.path.isfile(os.path.join(scoring_dir, "score.py")):
        print(f"ERROR: inference/cgcnn/score.py not found at {scoring_dir}")
        sys.exit(1)

    ml_client = build_client(args)

    # Step 1: endpoint
    scoring_uri = ensure_endpoint(ml_client, args.dry_run)

    if args.endpoint_only:
        print(f"\nEndpoint ready: {scoring_uri}")
        return

    # Step 2: filter deployments
    target = DEPLOYMENTS
    if args.only:
        target = [(n, m, p) for n, m, p in DEPLOYMENTS if n in args.only]
        unknown = set(args.only) - {n for n, _, _ in target}
        if unknown:
            print(f"WARNING: unknown deployment names: {unknown}")
            print(f"Valid names: {[n for n, _, _ in DEPLOYMENTS]}")

    print(f"\nDeploying {len(target)} model(s) to '{ENDPOINT_NAME}' ...")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Environment:   {args.environment}")
    print(f"  Scoring dir:   {scoring_dir}")

    succeeded, failed = [], []
    for deployment_name, model_ref, property_name in target:
        ok = deploy_one(
            ml_client=ml_client,
            deployment_name=deployment_name,
            model_ref=model_ref,
            property_name=property_name,
            scoring_dir=scoring_dir,
            instance_type=args.instance_type,
            environment=args.environment,
            dry_run=args.dry_run,
        )
        (succeeded if ok else failed).append(deployment_name)

    # Step 3: set default traffic to specific-heat if it was deployed
    if "specific-heat" in succeeded:
        set_default_traffic(ml_client, "specific-heat", args.dry_run)
    elif succeeded:
        set_default_traffic(ml_client, succeeded[0], args.dry_run)

    print_summary(ml_client if not args.dry_run else None, scoring_uri, succeeded, failed)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
